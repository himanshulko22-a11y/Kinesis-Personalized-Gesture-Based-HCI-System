"""Kinesis - personalized hand-gesture control for your computer.

Run:  python Kinesis_app.py

Files:
  config.py    settings and file paths
  actions.py   what a gesture can do (click, volume, scroll...)
  gestures.py  landmarks, dataset, model training, gesture -> action mapping
  widgets.py   small UI pieces (buttons, mapping row, confusion matrix)
  camera.py    webcam reader
  movements.py dynamic gestures (movements) with DTW
  style.py     window theme
"""
import os
import shutil
import sys
import threading
import time

import json

import cv2
import mediapipe as mp
import pandas as pd
import pyautogui
from PyQt5.QtCore import QRectF, Qt, QTimer, QUrl
from PyQt5.QtGui import (QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter, QPainterPath,
                         QPalette, QPixmap)
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QInputDialog, QKeySequenceEdit, QLabel, QLineEdit,
    QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

import config as cfg
import gestures as g
import actions
import movements as mv
from camera import Camera
from actions import (ACTIONS, SHORTCUT_PREFIX, add_shortcut, custom_shortcuts, hotkey_pressed, load_settings,
                     parse_hotkey, remove_shortcut, save_settings)
from style import STYLESHEET
from widgets import (CollapsibleSection, ConfusionMatrixDialog, GestureMappingRow, HandTile, SelectGesturesDialog,
                     ask, button, clear_layout, field_label, note)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
SCREEN_W, SCREEN_H = pyautogui.size()


# ---------- Main window ----------

class KinesisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinesis - Personalized Gesture Control")
        self.setGeometry(60, 40, 1320, 820)
        self.setMinimumSize(1000, 660)
        if os.path.exists(cfg.ICON_PATH):
            self.setWindowIcon(QIcon(cfg.ICON_PATH))

        load_error = self._load_model()

        self.camera = Camera()
        self.frame_id = 0
        self.timing = {"app_fps": 0.0, "tracking": 0.0, "gestures": 0.0, "display": 0.0}   # running averages
        self.last_processed = time.perf_counter()

        self.hand_states = {"Left": g.HandState(), "Right": g.HandState()}
        self.cursor = g.CursorSmoother()
        self.movements = mv.MovementLibrary()
        self.movement_trackers = {"Left": mv.MovementTracker(), "Right": mv.MovementTracker()}
        self.record_kind = "pose"           # what is being recorded: "pose" or "movement"
        self.record_tracker = mv.MovementTracker(need_rest=False)   # the countdown already gives a pause
        self.last_movement_time = {}
        self.recorded_path = None           # movement captured, waiting for Save / Retake
        self.countdown_end = 0.0
        self.trails = {}                    # hand -> finished movement shown on the video for a moment
        self.hand_view = {}                 # hand -> what its tile shows this frame (filled by the loop)
        self.status_pill_text = None
        self.cursor_used = False
        self.held_buttons = {}   # hand -> {"button", "gesture", "label", "kind"} for press-and-hold actions
        self.show_speed = True
        self.last_action_time = {}
        self.last_loop_error = None
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD
        self.cursor_sensitivity = cfg.CURSOR_SENSITIVITY
        self.paused = False
        self.zoom_enabled = False
        self.zoom_ctrl_held = False
        self.prev_hand_distance = None

        self.mode = "normal"   # normal | recording | reviewing | locked
        self.new_gesture_name = ""
        self.recording_buffer = []
        self.last_activity_time = time.time()
        self.last_frame_time = time.time()
        self.unlock_hold_start = None
        self.unlock_gesture = None
        self.last_fired = None        # last action that fired, so it can be marked as a mistake
        self.hotkey_down = False
        self.settings = load_settings()
        self.lite_tracking = self.settings.get("lite_tracking", cfg.LITE_TRACKING)
        self.hands = self._make_hand_tracker()
        self.training = None                # background training thread while it runs
        self.train_progress = 0.0
        self.train_result = None
        self.removed_poses = set()          # deleted poses the current model still knows (ignored until retrain)
        self.last_deleted = None            # what the Undo link would bring back
        for text in self.settings.get("shortcuts", []):
            try:
                add_shortcut(text)   # before the mapping loads, so gestures can keep these actions
            except ValueError:
                pass
        self.mistake_hotkey = self.settings.get("mistake_hotkey", cfg.MISTAKE_HOTKEY)
        self.show_trail = self.settings.get("show_trail", True)
        try:
            self.hotkey_codes = parse_hotkey(self.mistake_hotkey)
        except ValueError:
            self.mistake_hotkey, self.hotkey_codes = cfg.MISTAKE_HOTKEY, parse_hotkey(cfg.MISTAKE_HOTKEY)

        self._build_ui()
        try:
            self._clean_trash()   # drop anything deleted more than TRASH_KEEP_DAYS ago
        except (OSError, ValueError) as e:
            self._log(f"Trash clean-up skipped: {e}")
        if load_error:
            self.status_label.setText(f"Saved model couldn't be loaded ({load_error}). Retrain or Restore Defaults.")
        elif not self.camera.is_opened():
            self.status_label.setText("Camera not found - check it's connected and not used by another app.")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(cfg.FRAME_INTERVAL_MS)

    def _load_model(self):
        """Loads the saved model and mapping. Returns an error message if the files are broken."""
        try:
            self.model, self.encoder = g.load_trained_model()
            error = None
        except Exception as e:
            self.model, self.encoder, error = None, None, str(e)
        return error

    @staticmethod
    def _has_backup():
        return all(os.path.exists(backup) for current, backup in cfg.BACKUP_PAIRS
                   if current in (cfg.MODEL_PATH, cfg.ENCODER_PATH))

    # ---------- UI layout ----------

    def _build_ui(self):
        self.sections = {}
        central = QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(16, 16, 8, 16)
        outer.setSpacing(16)

        # Left: the camera, as big as the window allows (it keeps its shape)
        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # don't grow to the picture
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.addWidget(self.video_label)
        outer.addWidget(video_frame, 1)

        # Right: the control panel
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        col = QVBoxLayout(panel)
        col.setContentsMargins(4, 0, 12, 0)
        col.setSpacing(10)
        scroll = QScrollArea()
        scroll.setObjectName("panelScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        self.panel_scroll = scroll
        self.panel_layout = col
        right = QWidget()   # pinned top part (when locked) + the scrolling panel
        right.setMinimumWidth(500)
        right.setMaximumWidth(620)
        self.right_layout = QVBoxLayout(right)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(10)
        self.right_layout.addWidget(scroll, 1)
        outer.addWidget(right)

        # Top part: title, live card, pause/unlock. Pinned (doesn't scroll) unless the lock is opened.
        self.top_block = QWidget()
        top = QVBoxLayout(self.top_block)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("KINESIS")
        title.setObjectName("appTitle")
        font = title.font()
        font.setLetterSpacing(QFont.AbsoluteSpacing, 4)
        title.setFont(font)
        tagline = QLabel("Personalized gesture control")
        tagline.setObjectName("tagline")
        titles.addWidget(title)
        titles.addWidget(tagline)
        header.addLayout(titles)
        header.addStretch()
        self.pin_btn = QPushButton()
        self.pin_btn.setObjectName("pinBtn")
        self.pin_btn.setCursor(Qt.PointingHandCursor)
        self.pin_btn.setFixedSize(34, 30)
        self.pin_btn.clicked.connect(lambda: self._set_top_pinned(not self.top_pinned))
        header.addWidget(self.pin_btn, 0, Qt.AlignVCenter)
        self.status_pill = QLabel("")
        self.status_pill.setObjectName("statusPill")
        header.addWidget(self.status_pill, 0, Qt.AlignVCenter)
        top.addLayout(header)

        # Live card: one tile per hand, then the overall state and the last action
        live = QFrame()
        live.setObjectName("liveCard")
        live_layout = QVBoxLayout(live)
        live_layout.setContentsMargins(14, 12, 14, 12)
        live_layout.setSpacing(8)
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.hand_tiles = {"Left": HandTile("Left"), "Right": HandTile("Right")}
        for tile in self.hand_tiles.values():
            tiles.addWidget(tile)
        live_layout.addLayout(tiles)
        self.gesture_label = QLabel("No hand in view")
        self.gesture_label.setObjectName("gestureLabel")
        self.confidence_label = QLabel("")
        self.confidence_label.setObjectName("confidenceLabel")
        self.last_action_label = QLabel("Last action: -")
        self.last_action_label.setObjectName("lastActionLabel")
        self.last_action_label.setWordWrap(True)
        state_row = QHBoxLayout()
        state_row.addWidget(self.gesture_label)
        state_row.addStretch()
        state_row.addWidget(self.confidence_label)
        live_layout.addLayout(state_row)
        live_layout.addWidget(self.last_action_label)
        self.status_label = QLabel("")   # general messages: lock, training, camera problems...
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label, 1)
        self.undo_btn = button("Undo", self.undo_delete, "Bring back what was just deleted", name="linkBtn")
        self.undo_btn.setCursor(Qt.PointingHandCursor)
        self.undo_btn.hide()
        status_row.addWidget(self.undo_btn, 0, Qt.AlignTop)
        live_layout.addLayout(status_row)
        top.addWidget(live)

        row = QHBoxLayout()
        self.pause_btn = button("Pause Recognition", self.toggle_pause,
                                "Stops all actions. The camera keeps running.")
        row.addWidget(self.pause_btn)
        row.addWidget(button("Manual Unlock", lambda: self._unlock("manual button"),
                             f"The app locks after {cfg.IDLE_LOCK_TIMEOUT}s with no hand. Unlock here, "
                             f"or hold the unlock gesture for {cfg.UNLOCK_HOLD_SECONDS}s.", name="secondaryBtn"))
        top.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(field_label("Unlock gesture"))
        self.unlock_combo = QComboBox()
        self.unlock_combo.currentIndexChanged.connect(
            lambda: setattr(self, "unlock_gesture", self.unlock_combo.currentData()))
        row.addWidget(self.unlock_combo, 1)
        top.addLayout(row)
        self.top_pinned = None
        self._set_top_pinned(self.settings.get("pin_top", True), save=False)

        row = QHBoxLayout()
        row.addStretch()
        for text, opened in (("Expand all", True), ("Collapse all", False)):
            link = button(text, lambda opened=opened: self._expand_all(opened), name="linkBtn")
            link.setCursor(Qt.PointingHandCursor)
            row.addWidget(link)
        col.addLayout(row)

        box = self._section(col, "Gesture Mapping", True)
        self.mapping_layout = QVBoxLayout()
        self.mapping_layout.setAlignment(Qt.AlignTop)
        box.addLayout(self.mapping_layout)
        self.pending_label = QLabel("")
        self.pending_label.setObjectName("pendingLabel")
        self.pending_label.setWordWrap(True)
        self.pending_label.hide()
        box.addWidget(self.pending_label)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(button("Delete Several...", self.delete_several,
                             "Tick several gestures and delete them together. Deleting is instant - "
                             "retrain whenever it suits you.", name="deleteBtn"))
        box.addLayout(row)

        box = self._section(col, "Add New Gesture", True)
        row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. fist, or swipe_right")
        row.addWidget(self.name_input)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("Static pose", userData="pose")
        self.kind_combo.addItem("Movement", userData="movement")
        self.kind_combo.setToolTip("Static pose: a hand shape you hold (needs Retrain).\n"
                                   "Movement: a motion like a swipe or circle, made with an open palm "
                                   "(works straight away, no Retrain).")
        row.addWidget(self.kind_combo)
        box.addLayout(row)
        self.augment_box = QCheckBox("Train with data augmentation")
        self.augment_box.setChecked(cfg.TRAIN_WITH_AUGMENTATION)
        self.augment_box.setToolTip("Adds jittered and mirrored copies of each frame when training.")
        box.addWidget(self.augment_box)
        row = QHBoxLayout()
        self.record_btn = button("Record Gesture", self.start_recording)
        tips_btn = button("ⓘ", self.show_recording_tips, "Tips for recording a gesture")
        tips_btn.setFixedWidth(28)
        self.train_btn = button("Retrain Model", lambda: self._retrain("Retrained."))
        for widget in (self.record_btn, tips_btn, self.train_btn):
            row.addWidget(widget)
        box.addLayout(row)
        box.addWidget(button("Self-Test Accuracy", self.run_self_test,
                             "Tests the model on a random slice of your data. Some of it was used in "
                             "training, so treat it as a sanity check - live testing is the real test.",
                             name="secondaryBtn"))
        self.record_label = QLabel("")   # recording progress and results
        self.record_label.setObjectName("statusLabel")
        self.record_label.setWordWrap(True)
        box.addWidget(self.record_label)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        box.addWidget(self.warning_label)
        row = QHBoxLayout()
        self.save_btn = button("Save Gesture", self.save_recording)
        self.retake_btn = button("Retake", lambda: self._begin_recording(self.new_gesture_name), name="secondaryBtn")
        self.cancel_btn = button("Cancel", self.cancel_recording, name="deleteBtn")
        for widget in (self.save_btn, self.retake_btn, self.cancel_btn):
            row.addWidget(widget)
        box.addLayout(row)

        box = self._section(col, "Sensitivity", False)
        self._add_slider(box, "Confidence Threshold", 50, 95, int(self.confidence_threshold * 100),
                         lambda v: f"{v}%", lambda v: setattr(self, "confidence_threshold", v / 100))
        self._add_slider(box, "Cursor Sensitivity", 1, 100, self.cursor_sensitivity,
                         str, lambda v: setattr(self, "cursor_sensitivity", v),
                         "Higher = less hand movement to reach the screen edges (faster, less precise).")

        box = self._section(col, "Custom Shortcuts", False)
        box.addWidget(note("Add any keyboard shortcut (e.g. Ctrl+S, Win+E, F5) and it appears as an action "
                           "in Gesture Mapping. Ready-made ones like Play/Pause and Next Slide are already there."))
        row = QHBoxLayout()
        self.shortcut_edit = QKeySequenceEdit()
        self.shortcut_edit.setToolTip("Click here and press the shortcut you want.")
        row.addWidget(self.shortcut_edit)
        row.addWidget(button("Add Shortcut", self.add_custom_shortcut))
        box.addLayout(row)
        row = QHBoxLayout()
        self.shortcut_list = QListWidget()
        self.shortcut_list.setFixedHeight(70)
        row.addWidget(self.shortcut_list)
        row.addWidget(button("Remove", self.remove_custom_shortcut, name="deleteBtn"))
        box.addLayout(row)
        self._refresh_shortcut_list()

        box = self._section(col, "Display & Advanced", False)
        zoom_box = QCheckBox("Two hands apart/together = Zoom in/out")
        zoom_box.setToolTip("Works when both hands hold a gesture mapped to 'None' (e.g. open_palm). "
                            "Apart = Ctrl+scroll up, together = Ctrl+scroll down.")
        zoom_box.stateChanged.connect(self.on_zoom_toggled)
        box.addWidget(zoom_box)
        box.addWidget(note("Each hand is recognized separately, so the left hand can move the "
                           "cursor while the right hand clicks."))
        speed_box = QCheckBox("Show speed (FPS and timing) on video")
        speed_box.setChecked(self.show_speed)
        speed_box.stateChanged.connect(lambda state: setattr(self, "show_speed", state == Qt.Checked))
        box.addWidget(speed_box)
        trail_box = QCheckBox("Show movement trail on video")
        trail_box.setToolTip("Draws the path of an open-hand movement while you make it, then shows "
                             "what it was recognised as (green) or that it wasn't (grey).")
        trail_box.setChecked(self.show_trail)
        trail_box.stateChanged.connect(self.on_trail_toggled)
        box.addWidget(trail_box)
        lite_box = QCheckBox("Faster hand tracking (lite model)")
        lite_box.setToolTip("Uses MediaPipe's lighter hand model: less time per frame, so a higher FPS.\n"
                            "Landmarks come out slightly different from your recordings, so check your "
                            "gestures still work (re-record any that don't).")
        lite_box.setChecked(self.lite_tracking)
        lite_box.stateChanged.connect(self.on_lite_toggled)
        box.addWidget(lite_box)
        box.addWidget(button("Open Screenshots Folder", self.open_screenshots_folder))

        box = self._section(col, "Learning From Mistakes", False)
        box.addWidget(note("Did a gesture fire by accident? Mark it within a few seconds (button, the "
                           "hotkey below, or a gesture mapped to 'Mark Last Action as Mistake'). "
                           "That hand shape is saved as 'not a gesture', and the model learns it on the next Retrain."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Mistake hotkey:"))
        self.hotkey_edit = QKeySequenceEdit(QKeySequence(self.mistake_hotkey))
        self.hotkey_edit.setToolTip("Click here, press the key or combination you want, then click elsewhere. "
                                    "It works from any app.")
        self.hotkey_edit.editingFinished.connect(self.on_hotkey_changed)
        row.addWidget(self.hotkey_edit)
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button("That Was a Mistake", lambda: self.mark_mistake("button")))
        row.addWidget(button("Clear Saved Mistakes", self.clear_mistakes, name="deleteBtn"))
        box.addLayout(row)

        box = self._section(col, "Dataset Info", False)
        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.setAlignment(Qt.AlignTop)
        box.addLayout(self.dataset_layout)
        row = QHBoxLayout()
        row.addWidget(button("Refresh Info", self._refresh_dataset_info))
        row.addWidget(button("Balance Dataset", self.balance_dataset,
                             "Trims every gesture to the smallest gesture's frame count."))
        box.addLayout(row)

        box = self._section(col, "Activity Log", False)
        self.log_widget = QListWidget()
        self.log_widget.setFixedHeight(200)
        self.log_widget.setWordWrap(True)                     # long entries wrap onto more lines...
        self.log_widget.setResizeMode(QListView.Adjust)       # ...and re-wrap when the panel is resized
        self.log_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        box.addWidget(self.log_widget)
        box.addWidget(button("Clear Log", self.log_widget.clear))

        box = self._section(col, "Danger Zone", False, danger=True)
        box.addWidget(button("Reset All Data (delete everything)", self.reset_all_data, name="deleteBtn"))
        row = QHBoxLayout()
        row.addWidget(button("Save as Default", self.save_as_default,
                             "Saves your current gestures, model and mapping as the default backup."))
        self.restore_btn = button("Restore Defaults", self.restore_defaults,
                                  "Replaces your current data with the saved default backup.")
        row.addWidget(self.restore_btn)
        box.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button("Restore Deleted Gestures...", self.restore_deleted,
                             f"Bring back poses or movements you deleted (kept for {cfg.TRASH_KEEP_DAYS} days).",
                             name="secondaryBtn"))
        row.addWidget(button("Empty Trash", self.empty_trash,
                             "Permanently delete everything in the trash now.", name="deleteBtn"))
        box.addLayout(row)
        col.addStretch()

        self._set_recording_buttons()
        self._refresh_after_model_change()

    def _section(self, layout, title, default_open, danger=False):
        """Adds a collapsible section and returns the layout to put its contents in.
        Open/closed is remembered between runs."""
        opened = self.settings.get("sections", {}).get(title, default_open)
        section = CollapsibleSection(title, opened, self._on_section_toggled, danger)
        layout.addWidget(section)
        self.sections[title] = section
        return section.content

    def _set_top_pinned(self, pinned, save=True):
        """Locked: the top part stays in place while the sections scroll under it.
        Unlocked: it scrolls away with everything else."""
        if pinned == self.top_pinned:
            return
        self.top_pinned = pinned
        if pinned:
            self.panel_layout.removeWidget(self.top_block)
            self.top_block.setContentsMargins(4, 0, 22, 0)   # line up with the panel beside its scrollbar
            self.right_layout.insertWidget(0, self.top_block)
        else:
            self.right_layout.removeWidget(self.top_block)
            self.top_block.setContentsMargins(0, 0, 0, 0)
            self.panel_layout.insertWidget(0, self.top_block)
        self.pin_btn.setText("🔒" if pinned else "🔓")
        self.pin_btn.setToolTip("Top part is fixed - click to let it scroll" if pinned
                                else "Top part scrolls - click to fix it in place")
        if save:
            self.settings["pin_top"] = pinned
            save_settings(self.settings)

    def _on_section_toggled(self, title, opened):
        self.settings.setdefault("sections", {})[title] = opened
        save_settings(self.settings)

    def _expand_all(self, opened):
        for section in self.sections.values():
            section.set_expanded(opened)
        self.settings["sections"] = {title: opened for title in self.sections}
        save_settings(self.settings)

    def _add_slider(self, layout, title, low, high, value, fmt, on_change, tooltip=None):
        layout.addWidget(field_label(title))
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        if tooltip:
            slider.setToolTip(tooltip)
        value_label = QLabel(fmt(value))
        value_label.setFixedWidth(40)
        slider.valueChanged.connect(lambda v: (value_label.setText(fmt(v)), on_change(v)))
        row.addWidget(slider)
        row.addWidget(value_label)
        layout.addLayout(row)

    def _known_gestures(self):
        poses = list(self.encoder.classes_) if self.encoder is not None else []
        return poses + [m for m in self.movements.labels() if m not in poses]

    def _update_removed_poses(self):
        """Poses the model still knows but whose data was deleted: ignored until the next Retrain."""
        df = g.read_dataset()
        recorded = set(df["label"]) if df is not None else set()
        trained = set(self.encoder.classes_) if self.encoder is not None else set()
        self.removed_poses = trained - recorded - {g.NONE_LABEL}
        if self.removed_poses:
            self.pending_label.setText(f"{len(self.removed_poses)} deleted pose(s) are ignored for now - "
                                       "click Retrain Model when convenient to remove them from the model.")
            self.pending_label.show()
        else:
            self.pending_label.hide()

    def _refresh_after_model_change(self):
        self._update_removed_poses()
        self.gesture_mapping = g.load_mapping(self._known_gestures())
        self._clear_prediction_state()
        self.last_action_time = {}
        self._refresh_mapping_rows()
        self._refresh_dataset_info()
        self._refresh_unlock_options()
        self.restore_btn.setEnabled(self._has_backup())

    def _refresh_dataset_info(self):
        clear_layout(self.dataset_layout)
        self._show_movement_info()
        df = g.read_dataset()
        if df is None or df.empty:
            self.dataset_layout.addWidget(QLabel("No static poses recorded yet."))
            return
        counts = df["label"].value_counts()
        mistakes = int(counts.get(g.NONE_LABEL, 0))
        counts = counts.drop(g.NONE_LABEL, errors="ignore")
        total = QLabel(f"Total frames: {len(df)}")
        total.setStyleSheet("font-weight: bold; color: #c9a8ff;")
        self.dataset_layout.addWidget(total)
        if mistakes:
            used = "used in training" if mistakes >= cfg.MIN_MISTAKE_FRAMES else \
                f"used once there are {cfg.MIN_MISTAKE_FRAMES}"
            label = QLabel(f"not a gesture (from {mistakes // cfg.MISTAKE_FRAMES} corrected mistakes): "
                           f"{mistakes} frames, {used}")
            label.setStyleSheet("color: #9d8ec2;")
            self.dataset_layout.addWidget(label)
        average = counts.sum() / max(len(counts), 1)
        for gesture, count in counts.items():
            low = count < average * 0.5
            label = QLabel(f"{gesture}: {count} frames" + ("  (low - consider recording more)" if low else ""))
            label.setStyleSheet(f"color: {'#ffb08a' if low else '#e6def7'};")
            self.dataset_layout.addWidget(label)

    def _show_movement_info(self):
        counts = self.movements.counts()
        if not counts:
            return
        mistakes = counts.pop(mv.NONE_MOVEMENT, 0)
        header = QLabel(f"Movements: {sum(counts.values())} recordings")
        header.setStyleSheet("font-weight: bold; color: #c9a8ff;")
        self.dataset_layout.addWidget(header)
        for name, count in sorted(counts.items()):
            low = count < 3
            label = QLabel(f"{name}: {count} recordings" + ("  (record 4-5 for reliable recognition)" if low else ""))
            label.setStyleSheet(f"color: {'#ffb08a' if low else '#e6def7'};")
            self.dataset_layout.addWidget(label)
        if mistakes:
            label = QLabel(f"not a movement (corrected mistakes): {mistakes}")
            label.setStyleSheet("color: #9d8ec2;")
            self.dataset_layout.addWidget(label)

    def _refresh_mapping_rows(self):
        clear_layout(self.mapping_layout)
        poses = [p for p in (self.encoder.classes_ if self.encoder is not None else [])
                 if p != g.NONE_LABEL and p not in self.removed_poses]
        moves = self.movements.labels()
        if not poses and not moves:
            self.mapping_layout.addWidget(QLabel("No gestures trained yet."))
            return
        if poses:
            self.mapping_layout.addWidget(self._group_label(f"STATIC POSES  ·  {len(poses)}"))
        for gesture in poses:
            self.mapping_layout.addWidget(GestureMappingRow(
                gesture, self.gesture_mapping.get(gesture, "none"),
                self.on_mapping_changed, lambda name: self._begin_recording(name, "pose"), self.delete_gesture,
                on_rename=self.rename_gesture))
        if moves:
            self.mapping_layout.addWidget(self._group_label(
                f"MOVEMENTS  ·  {len(moves)}" + ("  ·  open hand" if cfg.MOVEMENT_NEEDS_OPEN_HAND else "")))
        for name in moves:
            self.mapping_layout.addWidget(GestureMappingRow(
                name, self.gesture_mapping.get(name, "none"), self.on_mapping_changed,
                lambda n: self._begin_recording(n, "movement"), self.delete_movement,
                on_rename=lambda n: self.rename_gesture(n, movement=True), movement=True))

    @staticmethod
    def _group_label(text):
        label = QLabel(text)
        label.setObjectName("groupLabel")
        return label

    def _refresh_unlock_options(self):
        combo = self.unlock_combo
        combo.blockSignals(True)
        combo.clear()
        for gesture in (self.encoder.classes_ if self.encoder is not None else []):
            if gesture != g.NONE_LABEL and gesture not in self.removed_poses:
                combo.addItem(gesture, userData=gesture)
        if combo.count() == 0:
            combo.addItem("(no gestures trained)", userData=None)
        for choice in (self.unlock_gesture, "open_palm"):
            if combo.findData(choice) >= 0:
                combo.setCurrentIndex(combo.findData(choice))
                break
        else:
            combo.setCurrentIndex(0)
        self.unlock_gesture = combo.currentData()
        combo.blockSignals(False)

    # ---------- Settings handlers ----------

    def on_mapping_changed(self, gesture, action_key):
        if action_key != "none":
            other = next((g2 for g2, a in self.gesture_mapping.items() if g2 != gesture and a == action_key), None)
            if other is not None:
                if not ask(self, "Action already in use",
                           f"'{ACTIONS[action_key].label}' is already hooked to '{other}'.\n\n"
                           f"Move it to '{gesture}'? '{other}' will be set to None."):
                    self._refresh_mapping_rows()
                    return
                self.gesture_mapping[other] = "none"
        self.gesture_mapping[gesture] = action_key
        g.save_mapping(self.gesture_mapping)
        self._clear_prediction_state()
        self._refresh_mapping_rows()
        self._log(f"Mapping updated: {gesture} -> {ACTIONS[action_key].label}")

    def on_zoom_toggled(self, state):
        self.zoom_enabled = state == Qt.Checked
        self._reset_zoom()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("Resume Recognition" if self.paused else "Pause Recognition")
        self._clear_prediction_state()
        self.last_activity_time = time.time()
        if not self.paused:
            self.status_label.setText("")
        self._log("Recognition paused" if self.paused else "Recognition resumed")

    def _clear_prediction_state(self):
        self._release_all_buttons()   # never leave a mouse button stuck down
        for state in self.hand_states.values():
            state.reset()
        for tracker in self.movement_trackers.values():
            tracker.reset()
        self.trails = {}
        self.cursor.reset()
        self._reset_zoom()

    # ---------- Lock ----------

    def _lock_message(self):
        if self.model is None or self.unlock_gesture is None:
            return "Locked - click Manual Unlock"
        return f"Locked - hold '{self.unlock_gesture}' or click Manual Unlock"

    def _enter_locked_mode(self):
        self.mode = "locked"
        self._clear_prediction_state()
        self.unlock_hold_start = None
        self.gesture_label.setText("Locked")
        self.confidence_label.setText("")
        self.status_label.setText(self._lock_message())
        self._log(f"Auto-lock (idle {cfg.IDLE_LOCK_TIMEOUT}s)")

    def _unlock(self, method):
        if self.mode != "locked":
            return
        self.mode = "normal"
        self.unlock_hold_start = None
        self.last_activity_time = time.time()
        self._clear_prediction_state()
        self.status_label.setText("")
        self._log(f"Unlocked ({method})")

    # ---------- Logging ----------

    def _log(self, message):
        entry = QListWidgetItem(f"[{time.strftime('%H:%M:%S')}] {message}")
        entry.setToolTip(entry.text())   # hover shows the full entry too
        self.log_widget.insertItem(0, entry)
        while self.log_widget.count() > cfg.LOG_MAX_ENTRIES:
            self.log_widget.takeItem(self.log_widget.count() - 1)
        try:
            if os.path.exists(cfg.LOG_PATH) and os.path.getsize(cfg.LOG_PATH) > cfg.LOG_MAX_FILE_BYTES:
                os.replace(cfg.LOG_PATH, cfg.LOG_PATH + ".1")
            with open(cfg.LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
        except OSError as e:
            print(f"Activity log write failed: {e}")

    def open_screenshots_folder(self):
        os.makedirs(cfg.SCREENSHOT_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(cfg.SCREENSHOT_DIR))

    # ---------- Recording ----------

    def show_recording_tips(self):
        QMessageBox.information(
            self, "Recording tips",
            "- Show only ONE hand while recording.\n"
            "- Move it slightly closer and farther, left and right, and tilt your wrist a little.\n"
            "- Keep the gesture shape the same - only position and angle should change.\n"
            f"- Keep your hand well lit and visible for all {cfg.FRAMES_PER_RECORDING} frames.\n\n"
            "This variety helps the model recognize the gesture wherever your hand is later.\n\n"
            "Movements (choose 'Movement' next to the name):\n"
            "- Use an open palm. After the 3-2-1 countdown, hold still, make the movement, then hold still again.\n"
            "- Record each movement 4-5 times: some faster, some slower, big and small, with either hand.\n"
            "- Direction matters: a swipe right is always a swipe right, whichever hand you use.\n"
            "- Movements work straight after saving - no Retrain needed.")

    def _set_recording_buttons(self):
        busy = self.mode in ("recording", "reviewing")
        self.record_btn.setEnabled(not busy)
        self.save_btn.setVisible(self.mode == "reviewing")
        self.retake_btn.setVisible(busy)
        self.cancel_btn.setVisible(busy)

    def start_recording(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Type a gesture name first.")
            return
        kind = self.kind_combo.currentData()
        problem = self._name_problem(name, kind, allow_existing=True)
        if problem:
            QMessageBox.warning(self, "Name not allowed", problem)
            return
        self._begin_recording(name, kind)

    def _name_problem(self, name, kind, allow_existing=False):
        """Why a gesture name can't be used, or None. allow_existing: the same kind may reuse it
        (recording more of an existing gesture)."""
        if not name:
            return "Type a name first."
        if name.startswith("_"):
            return "Gesture names can't start with '_'."
        df = g.read_dataset()
        poses = set(df["label"]) if df is not None else set()
        movements = set(self.movements.labels())
        if name in (movements if kind == "pose" else poses):
            other = "movement" if kind == "pose" else "static pose"
            return f"'{name}' is already a {other}. Pick another name."
        if not allow_existing and name in (poses if kind == "pose" else movements):
            return f"'{name}' already exists. Pick another name."
        return None

    def rename_gesture(self, old, movement=False):
        if not movement and self._busy_training():
            return
        new, ok = QInputDialog.getText(self, "Rename gesture", f"New name for '{old}':", text=old)
        new = new.strip()
        if not ok or new == old:
            return
        problem = self._name_problem(new, "movement" if movement else "pose")
        if problem:
            QMessageBox.warning(self, "Name not allowed", problem)
            return
        if movement:
            self.movements.rename(old, new)
        else:
            df = g.read_dataset()
            df.loc[df["label"] == old, "label"] = new
            df.to_csv(cfg.CSV_PATH, index=False)
            if self.encoder is not None and old in self.encoder.classes_:
                g.rename_label(self.encoder, old, new)   # the model keeps working, no retrain
        if old in self.gesture_mapping:
            self.gesture_mapping[new] = self.gesture_mapping.pop(old)
            g.save_mapping(self.gesture_mapping)
        if self.unlock_gesture == old:
            self.unlock_gesture = new
        self._clear_prediction_state()
        self.gesture_mapping = g.load_mapping(self._known_gestures())
        self._refresh_mapping_rows()
        self._refresh_dataset_info()
        self._refresh_unlock_options()
        self.status_label.setText(f"Renamed '{old}' to '{new}'.")
        self._log(f"Renamed {'movement' if movement else 'pose'} '{old}' -> '{new}'")

    def _begin_recording(self, name, kind=None):
        kind = kind or self.record_kind
        if self.mode in ("recording", "reviewing") and name != self.new_gesture_name:
            return   # finish or cancel the current recording first
        self.new_gesture_name = name
        self.record_kind = kind
        self.recording_buffer = []
        self.recorded_path = None
        self._clear_prediction_state()   # also releases a held Ctrl from zoom
        self.mode = "recording"
        self._set_recording_buttons()
        self.warning_label.hide()
        section = self.sections.get("Add New Gesture")
        if section is not None:   # recording may start from a + button with the section closed
            section.set_expanded(True)
            self.panel_scroll.ensureWidgetVisible(section)
        if kind == "movement":
            self.record_tracker.reset()
            self.countdown_end = time.time() + cfg.MOVE_COUNTDOWN
            self.record_label.setText(f"Movement '{name}': get your open palm ready...")
        else:
            self.record_label.setText(f"Recording '{name}'... hold your gesture with one hand.")

    def _record_movement_frame(self, hands, now, frame):
        if now < self.countdown_end:
            self._draw_big_text(frame, str(int(self.countdown_end - now) + 1))
            self.record_tracker.reset()
            return
        tracker = self.record_tracker
        if len(hands) != 1:
            movement = tracker.lost() if not hands else (tracker.finish() if tracker.active else None)
            if movement is None and not hands and tracker.missing <= cfg.MOVE_MAX_GAP:
                return   # hand briefly lost in motion blur - keep waiting
            if movement is None:
                if tracker.last_movement is not None:
                    self._log(f"Movement recording '{self.new_gesture_name}': not used - {tracker.last_problem}")
                    tracker.last_movement = None
                tracker.reset()
                self.record_label.setText("Show ONE hand for the movement." if hands else
                                          f"Movement '{self.new_gesture_name}': show your hand, then move.")
                return
        else:
            was_active = tracker.active
            movement = tracker.update(hands[0][1], None, now)
            self._draw_text(frame, f"Hand speed {tracker.speed:4.1f}  (movement starts above {cfg.MOVE_START_SPEED})",
                            (10, frame.shape[0] - 15))
            if tracker.active and not was_active:
                self._log(f"Movement recording '{self.new_gesture_name}': started (speed {tracker.speed:.1f})")
            if movement is None and tracker.last_movement is not None:
                self._log(f"Movement recording '{self.new_gesture_name}': not used - {tracker.last_problem}")
                self.record_label.setText(f"That movement was {tracker.last_problem} - try again, a bit bigger.")
                tracker.last_movement = None
                return
        if self.record_tracker.active:
            self._draw_path(frame, self.record_tracker.capture, (255, 200, 120))
            self.record_label.setText(f"Recording '{self.new_gesture_name}'... stop moving to finish.")
        elif movement is None:
            waited = now - self.countdown_end
            self.record_label.setText(
                f"Movement '{self.new_gesture_name}': GO - make the movement, then hold still."
                if waited < cfg.MOVE_RECORD_TIMEOUT else
                "No movement detected yet - move a bit faster, or press Retake / Cancel.")
        if movement is not None:
            self._log(f"Movement recording '{self.new_gesture_name}': captured {mv.describe(movement)}")
            self.recorded_path = movement
            self.mode = "reviewing"
            self._set_recording_buttons()
            seconds = movement[-1][3] - movement[0][3]
            self.record_label.setText(f"Captured '{self.new_gesture_name}' ({seconds:.1f} s). "
                                      f"Save Gesture to keep it, or Retake.")
            self._show_movement_warning()

    def _show_movement_warning(self):
        warning = self.movements.similarity_warning(self.new_gesture_name, self.recorded_path)
        if warning is None:
            self.warning_label.setText("✓ Looks different from your other movements.")
            self.warning_label.setStyleSheet("color: #7ee787;")
        else:
            level, message = warning
            self.warning_label.setText(("⚠ " if level == "strong" else "ℹ ") + message)
            self.warning_label.setStyleSheet(f"color: {'#ff9eb0' if level == 'strong' else '#ffb08a'};")
            self._log(f"Movement check for '{self.new_gesture_name}': {message}")
        self.warning_label.show()

    def _record_frame(self, hands):
        if len(hands) == 1:
            self.recording_buffer.append(g.normalize_landmarks(hands[0][1]))
            self.record_label.setText(
                f"Recording '{self.new_gesture_name}': {len(self.recording_buffer)}/{cfg.FRAMES_PER_RECORDING}")
        elif len(hands) > 1:
            self.record_label.setText("Show only ONE hand while recording.")
        if len(self.recording_buffer) >= cfg.FRAMES_PER_RECORDING:
            self.mode = "reviewing"
            self._set_recording_buttons()
            self.record_label.setText(
                f"Captured {len(self.recording_buffer)} frames for '{self.new_gesture_name}'. "
                f"Save Gesture to keep it, or Retake.")
            self._show_similarity_warning()

    def _show_similarity_warning(self):
        try:
            warning = g.similarity_warning(self.new_gesture_name, self.recording_buffer)
        except Exception as e:
            warning = None
            self._log(f"Similarity check failed: {e}")
        if warning is None:
            self.warning_label.setText("✓ Looks clearly different from your other gestures.")
            self.warning_label.setStyleSheet("color: #7ee787;")
        else:
            level, message = warning
            self.warning_label.setText(("⚠ " if level == "strong" else "ℹ ") + message)
            self.warning_label.setStyleSheet(f"color: {'#ff9eb0' if level == 'strong' else '#ffb08a'};")
            self._log(f"Similarity check for '{self.new_gesture_name}': {message}")
        self.warning_label.show()

    def save_recording(self):
        if self.record_kind == "movement":
            self._save_movement()
            return
        if self._busy_training():
            return   # the recording stays waiting for Save
        g.append_recording(self.new_gesture_name, self.recording_buffer)
        count = len(self.recording_buffer)
        self.recording_buffer = []
        self.mode = "normal"
        self._set_recording_buttons()
        self.warning_label.hide()
        self._refresh_dataset_info()
        self.record_label.setText(f"Saved {count} frames for '{self.new_gesture_name}'. "
                                  f"Record more takes for variety, or Retrain Model when ready.")
        self._log(f"Recorded {count} frames for '{self.new_gesture_name}'")

    def _save_movement(self):
        name = self.new_gesture_name
        self.movements.add(name, self.recorded_path)
        self.recorded_path = None
        self.mode = "normal"
        self._set_recording_buttons()
        self.warning_label.hide()
        self.gesture_mapping = g.load_mapping(self._known_gestures())
        self._refresh_mapping_rows()
        self._refresh_dataset_info()
        count = self.movements.counts().get(name, 0)
        self.record_label.setText(f"Saved movement {count} of '{name}'. "
                                  + ("Record 4-5 in total (press + next to it)." if count < 4 else
                                     "It works straight away - pick an action for it in Gesture Mapping."))
        self._log(f"Recorded movement '{name}' ({count} total)")

    def cancel_recording(self):
        self.recorded_path = None
        self.recording_buffer = []
        self.mode = "normal"
        self.last_activity_time = time.time()
        self._set_recording_buttons()
        self.warning_label.hide()
        self.record_label.setText("Recording cancelled.")

    # ---------- Learning from mistakes ----------

    def mark_mistake(self, source):
        """Saves the hand frames behind the last action as 'not a gesture'."""
        if self.training is not None:
            self.status_label.setText("Training in progress - mark the mistake again once it finishes.")
            return
        fired = self.last_fired
        if fired is None or time.time() - fired["time"] > cfg.MISTAKE_WINDOW:
            self.status_label.setText(f"No action in the last {cfg.MISTAKE_WINDOW:.0f} seconds to mark as a mistake.")
            return
        self.last_fired = None
        if fired.get("kind") == "movement":
            self.movements.add(mv.NONE_MOVEMENT, fired["path"])   # works straight away, no Retrain needed
            self.status_label.setText(f"Got it - that '{fired['gesture']}' movement was a mistake. "
                                      f"Kinesis will ignore movements like it from now on.")
            self._log(f"Mistake marked ({source}): {fired['hand']} hand movement {fired['gesture']} -> "
                      f"{fired['action']}, saved as not-a-movement")
            self._refresh_dataset_info()
            return
        g.append_recording(g.NONE_LABEL, fired["frames"])
        df = g.read_dataset()
        saved = int((df["label"] == g.NONE_LABEL).sum()) if df is not None else 0
        need = max(0, cfg.MIN_MISTAKE_FRAMES - saved)
        self.status_label.setText(
            f"Got it - that '{fired['gesture']}' was a mistake. " +
            (f"Mark {need // cfg.MISTAKE_FRAMES + (need % cfg.MISTAKE_FRAMES > 0)} more, then Retrain to learn them."
             if need else "Retrain Model to learn from your corrections."))
        self._log(f"Mistake marked ({source}): {fired['hand']} hand {fired['gesture']} -> {fired['action']}, "
                  f"{len(fired['frames'])} frames saved as not-a-gesture ({saved} total)")
        self._refresh_dataset_info()

    def clear_mistakes(self):
        if self._busy_training():
            return
        df = g.read_dataset()
        pose_mistakes = df is not None and (df["label"] == g.NONE_LABEL).any()
        movement_mistakes = mv.NONE_MOVEMENT in self.movements.counts()
        if not pose_mistakes and not movement_mistakes:
            QMessageBox.information(self, "No mistakes", "No corrected mistakes are saved.")
            return
        if not ask(self, "Clear saved mistakes?", "Delete everything saved from corrected mistakes?"):
            return
        if movement_mistakes:
            self.movements.remove(mv.NONE_MOVEMENT)
        if pose_mistakes:
            df[df["label"] != g.NONE_LABEL].to_csv(cfg.CSV_PATH, index=False)
        self._refresh_dataset_info()
        self.status_label.setText("Saved mistakes cleared. Retrain Model to update the model.")
        self._log("Cleared saved mistakes")

    # ---------- Custom shortcuts ----------

    def _refresh_shortcut_list(self):
        self.shortcut_list.clear()
        for text in custom_shortcuts():
            self.shortcut_list.addItem(text)
        if not self.shortcut_list.count():
            self.shortcut_list.addItem("(no custom shortcuts yet)")

    def _save_shortcuts(self):
        self.settings["shortcuts"] = custom_shortcuts()
        save_settings(self.settings)

    def add_custom_shortcut(self):
        text = self.shortcut_edit.keySequence().toString().split(",")[0].strip()
        self.shortcut_edit.clear()
        if not text:
            QMessageBox.information(self, "No shortcut", "Click the box, press a shortcut, then click Add Shortcut.")
            return
        if text in custom_shortcuts():
            QMessageBox.information(self, "Already added", f"'{text}' is already in your shortcuts.")
            return
        try:
            add_shortcut(text)
        except ValueError as e:
            QMessageBox.warning(self, "Shortcut not supported", str(e))
            return
        self._save_shortcuts()
        self._refresh_shortcut_list()
        self._refresh_mapping_rows()
        self.status_label.setText(f"Added shortcut {text} - pick it for a gesture in Gesture Mapping.")
        self._log(f"Custom shortcut added: {text}")

    def remove_custom_shortcut(self):
        item = self.shortcut_list.currentItem()
        if item is None or item.text() not in custom_shortcuts():
            QMessageBox.information(self, "Nothing selected", "Select a shortcut in the list first.")
            return
        text = item.text()
        users = [g2 for g2, a in self.gesture_mapping.items() if a == SHORTCUT_PREFIX + text]
        if users and not ask(self, "Remove shortcut?", f"'{text}' is used by {', '.join(users)}. "
                                                      f"Remove it and set those gestures to None?"):
            return
        remove_shortcut(text)
        for gesture in users:
            self.gesture_mapping[gesture] = "none"
        g.save_mapping(self.gesture_mapping)
        self._save_shortcuts()
        self._refresh_shortcut_list()
        self._refresh_mapping_rows()
        self._log(f"Custom shortcut removed: {text}")

    def on_hotkey_changed(self):
        text = self.hotkey_edit.keySequence().toString().split(",")[0].strip()
        if not text or text == self.mistake_hotkey:
            self.hotkey_edit.setKeySequence(QKeySequence(self.mistake_hotkey))
            return
        try:
            codes = parse_hotkey(text)
        except ValueError as e:
            QMessageBox.warning(self, "Hotkey not supported", str(e))
            self.hotkey_edit.setKeySequence(QKeySequence(self.mistake_hotkey))
            return
        self.mistake_hotkey, self.hotkey_codes = text, codes
        self.hotkey_down = True   # don't fire on the key press that set it
        self.settings["mistake_hotkey"] = text
        save_settings(self.settings)
        self.status_label.setText(f"Mistake hotkey is now {text}.")
        self._log(f"Mistake hotkey changed to {text}")

    def _check_mistake_hotkey(self):
        """The mistake hotkey works from any app (Windows only)."""
        if sys.platform != "win32" or self.hotkey_edit.hasFocus() or self.shortcut_edit.hasFocus():
            return   # not while the user is choosing a new hotkey
        down = hotkey_pressed(self.hotkey_codes)
        if down and not self.hotkey_down:
            self.mark_mistake(self.mistake_hotkey)
        self.hotkey_down = down

    # ---------- Training and data management ----------

    def _retrain(self, done_message):
        """Trains in a background thread: the camera and gestures keep working with the old model,
        and the new one is swapped in when it's ready (see _check_training)."""
        if self.training is not None:
            self.status_label.setText("Already training - please wait.")
            return
        if g.read_dataset() is None:
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return
        augment = self.augment_box.isChecked()   # read the checkbox here: Qt widgets belong to this thread
        self.train_progress, self.train_result = 0.0, None
        self.train_done_message = done_message
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training... 0%")
        self.status_label.setText("Training in the background - gestures keep working with the current model.")

        def work():
            try:
                self.train_result = ("ok", g.train_model(augment, progress=self._set_train_progress))
            except Exception as e:   # reported on the main thread
                self.train_result = ("error", e)

        self.training = threading.Thread(target=work, daemon=True)
        self.training.start()

    def _set_train_progress(self, fraction):
        self.train_progress = fraction   # just a number; the main thread shows it

    def _check_training(self):
        """Runs every frame on the main thread: shows progress, and swaps the new model in when done."""
        if self.training is None:
            return
        if self.training.is_alive():
            self.train_btn.setText(f"Training... {self.train_progress:.0%}")
            return
        self.training = None
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Retrain Model")
        status, result = self.train_result or ("error", RuntimeError("training stopped unexpectedly"))
        if status == "error":
            self.status_label.setText(f"Training failed: {result}")
            self._log(f"Retraining FAILED: {result}")
            return
        self.model, self.encoder, accuracy, raw, used = result
        self._refresh_after_model_change()
        frames = f"{raw} frames -> {used} after augmentation" if used != raw else f"{raw} frames"
        message = self.train_done_message
        self.status_label.setText(f"{message} {len(self.encoder.classes_)} gestures, {frames}, "
                                  f"test accuracy {accuracy:.0%} (often optimistic - trust live testing).")
        self._log(f"{message} test accuracy {accuracy:.1%}, {frames}")

    def _busy_training(self):
        """True (with a message) if the gesture data mustn't change right now because training is reading it."""
        if self.training is None:
            return False
        QMessageBox.information(self, "Training in progress",
                                "The model is training in the background. Please wait a few seconds until "
                                "it finishes, then try again.")
        return True

    def balance_dataset(self):
        if self._busy_training():
            return
        df = g.read_dataset()
        mistakes = df[df["label"] == g.NONE_LABEL] if df is not None else None
        df = df[df["label"] != g.NONE_LABEL] if df is not None else None   # saved mistakes are kept as they are
        if df is None or df["label"].nunique() < 2:
            QMessageBox.warning(self, "Not enough data", "Record at least 2 gestures first.")
            return
        counts = df["label"].value_counts()
        smallest = counts.min()
        if smallest == counts.max():
            QMessageBox.information(self, "Already balanced", "All gestures already have the same frame count.")
            return
        if not ask(self, "Balance dataset?", f"Trim every gesture down to {smallest} frames? "
                                             f"Extra frames are permanently deleted."):
            return
        balanced = df.groupby("label").sample(n=smallest, random_state=42)
        pd.concat([balanced, mistakes]).to_csv(cfg.CSV_PATH, index=False)
        self._retrain(f"Balanced to {smallest} frames per gesture and retrained.")

    def _current_poses(self):
        """Poses the user has: recorded or in the model, minus deleted ones."""
        df = g.read_dataset()
        recorded = set(df["label"]) if df is not None else set()
        trained = set(self.encoder.classes_) if self.encoder is not None else set()
        return sorted((recorded | trained) - self.removed_poses - {g.NONE_LABEL})

    def delete_gesture(self, gesture):
        self._delete([gesture], [], ask_first=f"Delete the pose '{gesture}'?")

    def delete_movement(self, name):
        self._delete([], [name], ask_first=f"Delete all recordings of the movement '{name}'?")

    def delete_several(self):
        dialog = SelectGesturesDialog(self, self._current_poses(), self.movements.labels())
        if dialog.exec_():
            chosen = dialog.selected()
            self._delete([n for n, kind in chosen if kind == "pose"], [n for n, kind in chosen if kind == "movement"],
                         ask_first=None)

    def _delete(self, poses, moves, ask_first):
        """Deletes instantly - no retraining. Poses vanish from the list and are ignored by the current
        model until the next Retrain. Everything goes to a trash file, so it can be undone or restored."""
        if not poses and not moves or self._busy_training():
            return
        if poses and len(set(self._current_poses()) - set(poses)) < 2:
            QMessageBox.warning(self, "Cannot delete", "At least 2 static poses must stay so the model can be "
                                                       "trained. Record another pose first, or use Reset All Data.")
            return
        names = ", ".join(poses + moves)
        question = ask_first or f"Delete {len(poses) + len(moves)} gesture(s)?\n\n{names}"
        if not ask(self, "Delete?", question + "\n\nYou can undo this straight after, or restore it later."):
            return
        if poses:
            df = g.read_dataset()
            gone = df["label"].isin(poses)
            self._append_trash_rows(df[gone])
            df[~gone].to_csv(cfg.CSV_PATH, index=False)
        if moves:
            trash = self._load_movement_trash()
            trash += [{"label": label, "path": [list(p) for p in path], "deleted_at": time.time()}
                      for label, path in self.movements.samples if label in moves]
            self._save_movement_trash(trash)
            for name in moves:
                self.movements.remove(name)
        self.last_deleted = {"poses": poses, "moves": moves, "time": time.time()}
        self._log(f"Deleted {names}")
        self._refresh_after_model_change()
        self.status_label.setText(f"Deleted {names}.")
        self.undo_btn.show()

    # Trash: deleted poses (CSV rows) and movements (paths), kept until restored or Reset All Data
    @staticmethod
    def _read_pose_trash():
        """Deleted pose frames, each with the time it was deleted (deleted_at)."""
        if not os.path.exists(cfg.TRASH_CSV_PATH):
            return None
        trash = pd.read_csv(cfg.TRASH_CSV_PATH)
        if "deleted_at" not in trash:
            trash["deleted_at"] = time.time()   # from before deletion times were saved: count from now
        trash["deleted_at"] = trash["deleted_at"].fillna(time.time())
        return trash

    @classmethod
    def _append_trash_rows(cls, rows):
        trash = cls._read_pose_trash()
        rows = rows.assign(deleted_at=time.time())
        (rows if trash is None else pd.concat([trash, rows])).to_csv(cfg.TRASH_CSV_PATH, index=False)

    @staticmethod
    def _load_movement_trash():
        try:
            with open(cfg.TRASH_MOVEMENT_PATH) as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    @staticmethod
    def _save_movement_trash(trash):
        with open(cfg.TRASH_MOVEMENT_PATH, "w") as f:
            json.dump(trash, f)

    def _trash_contents(self):
        poses = []
        if os.path.exists(cfg.TRASH_CSV_PATH):
            poses = sorted(pd.read_csv(cfg.TRASH_CSV_PATH)["label"].unique())
        moves = sorted({item["label"] for item in self._load_movement_trash()})
        return poses, moves

    def _clean_trash(self):
        """Permanently removes gestures that have been in the trash longer than TRASH_KEEP_DAYS."""
        cutoff = time.time() - cfg.TRASH_KEEP_DAYS * 24 * 3600
        removed = set()
        trash = self._read_pose_trash()
        if trash is not None:
            old = trash["deleted_at"] < cutoff
            removed |= set(trash.loc[old, "label"])
            if old.all():
                os.remove(cfg.TRASH_CSV_PATH)
            elif old.any():
                trash[~old].to_csv(cfg.TRASH_CSV_PATH, index=False)
        moves = self._load_movement_trash()
        keep = [item for item in moves if item.get("deleted_at", time.time()) >= cutoff]
        if len(keep) != len(moves):
            removed |= {item["label"] for item in moves if item not in keep}
            self._save_movement_trash(keep)
        if removed:
            self._log(f"Trash: permanently removed {', '.join(sorted(removed))} "
                      f"(deleted more than {cfg.TRASH_KEEP_DAYS} days ago)")

    def empty_trash(self):
        poses, moves = self._trash_contents()
        if not poses and not moves:
            QMessageBox.information(self, "Trash is empty", "There are no deleted gestures.")
            return
        names = ", ".join(poses + moves)
        if not ask(self, "Empty trash?", f"Permanently delete these deleted gestures? They can't be restored "
                                         f"after this.\n\n{names}"):
            return
        for path in (cfg.TRASH_CSV_PATH, cfg.TRASH_MOVEMENT_PATH):
            if os.path.exists(path):
                os.remove(path)
        self.last_deleted = None   # nothing left to undo
        self.undo_btn.hide()
        self._log(f"Emptied trash: {names}")
        self.status_label.setText("Trash emptied.")

    def undo_delete(self):
        if self.last_deleted is not None:
            self._restore(self.last_deleted["poses"], self.last_deleted["moves"])
        self.last_deleted = None
        self.undo_btn.hide()

    def restore_deleted(self):
        poses, moves = self._trash_contents()
        if not poses and not moves:
            QMessageBox.information(self, "Nothing to restore", "There are no deleted gestures.")
            return
        dialog = SelectGesturesDialog(self, poses, moves, restore=True)
        if dialog.exec_():
            chosen = dialog.selected()
            self._restore([n for n, kind in chosen if kind == "pose"], [n for n, kind in chosen if kind == "movement"])

    def _restore(self, poses, moves):
        """Brings deleted gestures back from the trash."""
        if self._busy_training():
            return
        taken = set(self._current_poses()) | set(self.movements.labels())
        clash = sorted(set(poses + moves) & taken)
        if clash:
            QMessageBox.warning(self, "Name in use", f"You already have a gesture called {', '.join(clash)}. "
                                                     "Rename or delete it first, then restore.")
            return
        needs_retrain = []
        if poses and os.path.exists(cfg.TRASH_CSV_PATH):
            trash = self._read_pose_trash()
            back = trash["label"].isin(poses)
            rows = trash[back].drop(columns="deleted_at")
            df = g.read_dataset()
            pd.concat([df, rows] if df is not None else [rows]).to_csv(cfg.CSV_PATH, index=False)
            if back.all():
                os.remove(cfg.TRASH_CSV_PATH)
            else:
                trash[~back].to_csv(cfg.TRASH_CSV_PATH, index=False)
            known = set(self.encoder.classes_) if self.encoder is not None else set()
            needs_retrain = [p for p in poses if p not in known]
        if moves:
            trash = self._load_movement_trash()
            for item in trash:
                if item["label"] in moves:
                    self.movements.add(item["label"], item["path"])
            self._save_movement_trash([item for item in trash if item["label"] not in moves])
        names = ", ".join(poses + moves)
        self._log(f"Restored {names}")
        self._refresh_after_model_change()
        self.status_label.setText(f"Restored {names}." + (
            f" Click Retrain Model to use {', '.join(needs_retrain)} again." if needs_retrain else ""))

    def run_self_test(self):
        if self.model is None:
            QMessageBox.warning(self, "No model", "Record at least 2 gestures and Retrain Model first.")
            return
        df = g.read_dataset()
        if df is None:
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        index = {name: i for i, name in enumerate(self.encoder.classes_)}
        untrained = sorted(set(df["label"]) - set(index) - {g.NONE_LABEL})
        if untrained:
            QMessageBox.information(self, "Not trained yet", f"Not in the model yet: {', '.join(untrained)}.\n"
                                    "They're left out of this test - click Retrain Model to include them.")
        df = df[df["label"].isin(index)]   # skip anything the model wasn't trained on (e.g. few mistakes)
        try:
            y = df["label"].map(index).values
            _, X_test, _, y_test = train_test_split(df.drop("label", axis=1).values, y,
                                                    test_size=0.2, stratify=y)
        except ValueError:
            QMessageBox.warning(self, "Model out of date",
                                "Your data has changed since the last training. Click Retrain Model first.")
            return

        predicted = self.model.predict(X_test, verbose=0).argmax(axis=1)
        lines = [f"Overall accuracy: {(predicted == y_test).mean():.1%}  ({len(y_test)} samples)", ""]
        for i, name in enumerate(self.encoder.classes_):
            mask = y_test == i
            if mask.any():
                lines.append(f"  {name}: {(predicted[mask] == i).mean():.1%}  ({mask.sum()} samples)")
        self._log(f"Self-test - overall accuracy {(predicted == y_test).mean():.1%}")
        QMessageBox.information(self, "Self-Test Results", "\n".join(lines))
        matrix = confusion_matrix(y_test, predicted, labels=range(len(self.encoder.classes_)))
        ConfusionMatrixDialog(self.encoder.classes_, matrix, parent=self).exec_()

    def _remove_data_files(self):
        for path in cfg.DATA_FILES:
            if os.path.exists(path):
                os.remove(path)

    def reset_all_data(self):
        if self._busy_training():
            return
        if not ask(self, "Reset everything?",
                   "Permanently delete all recorded gestures, the trained model and your mapping?\n\n"
                   "Your saved defaults are kept - use Restore Defaults to get them back."):
            return
        self._remove_data_files()
        for path in (cfg.TRASH_CSV_PATH, cfg.TRASH_MOVEMENT_PATH):
            if os.path.exists(path):
                os.remove(path)
        self.last_deleted = None
        self.undo_btn.hide()
        self.movements.load()
        self.model = self.encoder = None
        self.mode = "normal"
        self.recording_buffer = []
        self.recorded_path = None
        self.warning_label.hide()
        self._set_recording_buttons()
        self._refresh_after_model_change()
        self.gesture_label.setText("No hand in view")
        self.confidence_label.setText("")
        self.last_action_label.setText("Last action: -")
        self.status_label.setText("Everything reset. Record new gestures, or Restore Defaults.")
        self._log("Full reset: all data, model, and mapping deleted")

    def save_as_default(self):
        if self._busy_training():
            return
        if self.model is None:
            QMessageBox.warning(self, "No model", "Train a model first, then save it as the default.")
            return
        if not ask(self, "Save as default?", "Replace the saved default with your current gestures, model and mapping?"):
            return
        for current, backup in cfg.BACKUP_PAIRS:
            if os.path.exists(current):
                shutil.copy2(current, backup)
        self.restore_btn.setEnabled(True)
        self.status_label.setText("Current setup saved as the default.")
        self._log("Saved current setup as default")

    def restore_defaults(self):
        if self._busy_training():
            return
        if not self._has_backup():
            QMessageBox.information(self, "No defaults", "No saved default found. Use Save as Default first.")
            self.restore_btn.setEnabled(False)
            return
        if not ask(self, "Restore defaults?", "Replace your current gestures, model and mapping with the saved default?"):
            return
        self._remove_data_files()
        for current, backup in cfg.BACKUP_PAIRS:
            if os.path.exists(backup):
                shutil.copy2(backup, current)
        error = self._load_model()
        self.movements.load()
        self._refresh_after_model_change()
        if error or self.model is None:
            self.status_label.setText(f"Restore failed: {error or 'backup model missing'}")
            self._log(f"Restore FAILED: {error}")
        else:
            self.status_label.setText(f"Restored defaults - {len(self.encoder.classes_)} gestures loaded.")
            self._log("Restored gestures/model from default backup")

    # ---------- Main loop (runs every frame) ----------

    def update_frame(self):
        try:
            self._process_frame()
        except Exception as e:   # an error in a Qt timer would otherwise close the whole app
            self._release_zoom_ctrl()
            self._release_all_buttons()
            if str(e) != self.last_loop_error:
                self.last_loop_error = str(e)
                self._log(f"Error: {e}")

    def _process_frame(self):
        while self.camera.messages:
            self._log(self.camera.messages.pop(0))
        while actions.messages:
            self._log(actions.messages.pop(0))
        self._check_mistake_hotkey()
        self._check_training()
        frame, self.frame_id = self.camera.latest(self.frame_id)
        if frame is None:   # no new frame yet
            if time.time() - self.camera.last_frame_time > 3 and self.camera.is_opened():
                self.status_label.setText("Camera not returning frames - check it's connected and not used by another app.")
            return

        t0 = time.perf_counter()
        self.hand_view = {}
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = self._find_hands(self.hands.process(rgb), frame)
        t1 = time.perf_counter()

        now = time.time()
        if now - self.last_frame_time > 2.0 or self.paused or self.mode in ("recording", "reviewing"):
            self.last_activity_time = now   # don't count pauses, training or recording as idle time
        self.last_frame_time = now

        if self.mode == "recording" and self.record_kind == "movement":
            self._record_movement_frame(hands, now, frame)
        elif self.mode == "recording":
            self._record_frame(hands)
        elif self.mode == "reviewing" and self.recorded_path:
            self._draw_path(frame, self.recorded_path, (120, 255, 160))
        elif self.mode == "locked":
            self._update_locked(hands)
        elif self.mode == "normal":
            self._update_normal(hands, now)
            if self.show_trail:
                self._draw_trails(frame, now)
        t2 = time.perf_counter()

        if self.show_speed:
            self._draw_speed(frame)
        self._render_live(now)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        picture = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
        self.video_label.setPixmap(self._rounded(picture, 14))
        t3 = time.perf_counter()
        self._update_timing(t0, t1, t2, t3)

    @staticmethod
    def _rounded(picture, radius):
        """The video with rounded corners."""
        result = QPixmap(picture.size())
        result.fill(Qt.transparent)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(result.rect()), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, picture)
        painter.end()
        return result

    def _render_live(self, now):
        """Updates the hand tiles and the status pill from this frame's results."""
        if self.last_deleted is not None and time.time() - self.last_deleted["time"] > cfg.UNDO_SECONDS:
            self.last_deleted = None
            self.undo_btn.hide()
        for label, tile in self.hand_tiles.items():
            trail = self.trails.get(label)
            if trail is not None and trail["text"] != "not recognised" and now <= trail["until"]:
                action = ACTIONS[self.gesture_mapping.get(trail["text"], "none")]
                tile.show_hand("↝ " + trail["text"], 1.0,
                               f"→ {action.label}" if action.fn is not None else "movement", "move")
            elif label in self.hand_view:
                tile.show_hand(*self.hand_view[label])
            else:
                tile.show_hand(None, 0.0, "not in view", None)
        if self.mode == "recording":
            state, text = "rec", "●  REC"
        elif self.mode == "reviewing":
            state, text = "review", "REVIEW"
        elif self.mode == "locked":
            state, text = "locked", "LOCKED"
        elif self.paused:
            state, text = "paused", "❚❚  PAUSED"
        else:
            state, text = "live", "●  LIVE"
        if text != self.status_pill_text:
            self.status_pill_text = text
            self.status_pill.setText(text)
            self.status_pill.setProperty("state", state)
            self.status_pill.style().unpolish(self.status_pill)
            self.status_pill.style().polish(self.status_pill)

    def _set_hand_view(self, label, gesture, confidence, action_key):
        """What a hand's tile shows: its gesture, confidence and the action it runs."""
        if gesture is None:
            self.hand_view[label] = (None, confidence, "unsure", "low")
            return
        shown = "not a gesture" if gesture == g.NONE_LABEL else gesture
        action = ACTIONS.get(action_key or "none", ACTIONS["none"])
        if self.paused:
            detail = "paused - no action"
        elif action_key in (None, "none"):
            detail = "no action"
        else:
            detail = f"→ {action.label}"
        level = "good" if confidence >= self.confidence_threshold else "low"
        self.hand_view[label] = (shown, confidence, detail, level)

    def _make_hand_tracker(self):
        """MediaPipe Hands. The lite model (complexity 0) is faster; the full one (1) is a little more precise."""
        return mp_hands.Hands(max_num_hands=2, model_complexity=0 if self.lite_tracking else 1,
                              min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def on_lite_toggled(self, state):
        self.lite_tracking = state == Qt.Checked
        self.hands.close()
        self.hands = self._make_hand_tracker()
        self._clear_prediction_state()
        self.settings["lite_tracking"] = self.lite_tracking
        save_settings(self.settings)
        self._log(f"Hand tracking model: {'lite (faster)' if self.lite_tracking else 'full'}")
        self.status_label.setText("Lite hand tracking on - compare 'Hand tracking ms' in the speed readout, "
                                  "and check your gestures still work well." if self.lite_tracking
                                  else "Full hand tracking on.")

    def on_trail_toggled(self, state):
        self.show_trail = state == Qt.Checked
        if not self.show_trail:
            self.trails = {}
        self.settings["show_trail"] = self.show_trail
        save_settings(self.settings)

    def _set_trail(self, label, movement, name, now):
        """Keeps a finished movement on screen for a moment, with what it was recognised as."""
        if self.show_trail:
            color, text = ((120, 255, 160), name) if name else ((170, 170, 170), "not recognised")
            self.trails[label] = {"points": movement, "color": color, "text": text,
                                  "until": now + cfg.TRAIL_SECONDS}

    def _draw_trails(self, frame, now):
        """Live trail of a movement being made (orange), then the finished one for a moment."""
        h, w = frame.shape[:2]
        for label, tracker in self.movement_trackers.items():
            capture = tracker.capture
            if capture and len(capture) > 1 and mv.open_fraction(capture) >= cfg.MOVE_OPEN_SHARE:
                self._draw_path(frame, capture, (60, 170, 255), taper=True)
        for label, trail in list(self.trails.items()):
            if now > trail["until"]:
                del self.trails[label]
                continue
            self._draw_path(frame, trail["points"], trail["color"], taper=True)
            x, y = trail["points"][-1][0], trail["points"][-1][1]
            px, py = int(x * w), int(y / mv.ASPECT * h)
            cv2.putText(frame, trail["text"], (max(px - 40, 5), max(py - 20, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, trail["color"], 2, cv2.LINE_AA)

    def _draw_path(self, frame, points, color, taper=False):
        """Draws a palm path (from movements.hand_point coordinates) on the video."""
        h, w = frame.shape[:2]
        pixels = [(int(p[0] * w), int(p[1] / mv.ASPECT * h)) for p in points]
        for i, (a, b) in enumerate(zip(pixels, pixels[1:])):
            thickness = 2 + int(6 * (i + 1) / len(pixels)) if taper else 3   # thin tail, thick head
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)
        if pixels:
            cv2.circle(frame, pixels[0], 6, (255, 255, 255), -1)   # start point

    def _draw_text(self, frame, text, position):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1, cv2.LINE_AA)

    def _draw_big_text(self, frame, text):
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 8)
        cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 4,
                    (201, 168, 255), 8, cv2.LINE_AA)

    def _update_timing(self, t0, t1, t2, t3):
        """Running averages of how long each step takes, in milliseconds."""
        smooth = lambda old, new: 0.9 * old + 0.1 * new
        timing = self.timing
        timing["app_fps"] = smooth(timing["app_fps"], 1 / max(t3 - self.last_processed, 1e-3))
        timing["tracking"] = smooth(timing["tracking"], (t1 - t0) * 1000)
        timing["gestures"] = smooth(timing["gestures"], (t2 - t1) * 1000)
        timing["display"] = smooth(timing["display"], (t3 - t2) * 1000)
        self.last_processed = t3

    def _draw_speed(self, frame):
        t = self.timing
        lines = [f"Camera {self.camera.fps:4.1f} fps ({self.camera.backend})   App {t['app_fps']:4.1f} fps",
                 f"Hand tracking {t['tracking']:4.1f} ms",
                 f"Gestures+actions {t['gestures']:4.1f} ms",
                 f"Display {t['display']:4.1f} ms"]
        box = frame[6:92, 4:345]
        box[:] = (box * 0.35).astype(box.dtype)   # dark backdrop so the numbers are readable
        for i, text in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)

    def _find_hands(self, results, frame):
        """Returns [(label, landmarks)] and draws the hands on the frame."""
        hands = []
        handedness = results.multi_handedness or []
        for i, landmarks in enumerate(results.multi_hand_landmarks or []):
            label = handedness[i].classification[0].label if i < len(handedness) else None
            hands.append([label, landmarks])
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())
        # MediaPipe sometimes gives both hands the same label; then use screen side instead.
        if len(hands) == 2 and (hands[0][0] == hands[1][0] or None in (hands[0][0], hands[1][0])):
            hands.sort(key=lambda h: h[1].landmark[0].x)
            hands[0][0], hands[1][0] = "Left", "Right"
        elif len(hands) == 1 and hands[0][0] is None:
            hands[0][0] = "Right"
        return [tuple(h) for h in hands]

    def _update_locked(self, hands):
        self.gesture_label.setText("Locked")
        if self.model is None or self.unlock_gesture is None or not hands:
            self.unlock_hold_start = None
            self.confidence_label.setText("")
            self.status_label.setText(self._lock_message())
            return
        matches = [conf for gesture, conf in (g.predict(self.model, self.encoder, lm) for _, lm in hands)
                   if gesture == self.unlock_gesture and conf >= self.confidence_threshold]
        if not matches:
            self.unlock_hold_start = None
            self.confidence_label.setText("")
            self.status_label.setText(self._lock_message())
            return
        if self.unlock_hold_start is None:
            self.unlock_hold_start = time.time()
        held = time.time() - self.unlock_hold_start
        self.confidence_label.setText(f"Confidence: {max(matches):.0%}")
        self.status_label.setText(f"Hold '{self.unlock_gesture}' to unlock: "
                                  f"{min(100, int(held / cfg.UNLOCK_HOLD_SECONDS * 100))}%")
        if held >= cfg.UNLOCK_HOLD_SECONDS:
            self._unlock(f"gesture ({self.unlock_gesture})")

    def _update_normal(self, hands, now):
        if self.paused and self.model is not None:
            self._show_gestures_only(hands)   # paused: still name the gestures, but do nothing
            return
        if self.paused or QApplication.activeModalWidget() is not None:
            # Paused, or a dialog is open (a gesture click must never press "Yes" on Reset All Data).
            self._clear_prediction_state()
            self.gesture_label.setText("Paused" if self.paused else "Dialog open - gestures off")
            self.confidence_label.setText("")
            return
        if self.model is None:
            self.gesture_label.setText("No poses trained yet")
            self._update_movements(hands, {}, now)   # movements don't need the pose model
            return

        if hands:
            self.last_activity_time = now
            if self.status_label.text().startswith("No hand detected"):
                self.status_label.setText("")
        else:
            remaining = cfg.IDLE_LOCK_TIMEOUT - (now - self.last_activity_time)
            if remaining <= 0:
                self._enter_locked_mode()
                return
            if remaining <= cfg.IDLE_LOCK_WARNING:
                self.status_label.setText(f"No hand detected - locking in {int(remaining) + 1}s...")
            self.gesture_label.setText("No hand in view")
            self.confidence_label.setText("")

        action_keys = {}
        self.cursor_used = False
        for label, landmarks in hands:
            gesture, confidence, action_key = self._process_hand(label, landmarks)
            action_keys[label] = action_key
            self._set_hand_view(label, gesture, confidence, action_key)
        for label, state in self.hand_states.items():
            if label not in action_keys:
                state.push(None)
        self._release_finished_holds()
        if not self.cursor_used:
            self.cursor.reset()   # next time the cursor starts fresh from the hand's position
        self._update_movements(hands, action_keys, now)
        if hands:
            self.gesture_label.setText("Recognising")
            self.confidence_label.setText(f"{len(hands)} hand{'s' if len(hands) > 1 else ''} in view")

        both_unhooked = len(hands) == 2 and all(action_keys.get(label) == "none" for label, _ in hands)
        if self.zoom_enabled and both_unhooked:
            self._apply_zoom(hands[0][1], hands[1][1])
        else:
            self._reset_zoom()

    def _update_movements(self, hands, action_keys, now):
        """Feeds each hand to its movement tracker and runs the action of a recognised movement."""
        zooming = self.zoom_enabled and len(hands) == 2
        if not self.movements.labels() or zooming:
            for tracker in self.movement_trackers.values():
                tracker.reset()
            return
        seen = {label for label, _ in hands}
        for label, tracker in self.movement_trackers.items():
            if label not in seen:
                movement = tracker.lost()   # hand briefly lost (motion blur): wait a few frames
                if movement is not None:
                    self._handle_movement(label, movement, now)
                else:
                    self._log_dropped(label, tracker)
        for label, landmarks in hands:
            if action_keys.get(label) is not None and ACTIONS[action_keys[label]].kind in ("cursor", "hold", "hold_move"):
                self.movement_trackers[label].reset()   # steering the cursor or dragging is not a movement gesture
                continue
            tracker = self.movement_trackers[label]
            movement = tracker.update(landmarks, mv.is_open_hand(landmarks), now)
            if movement is not None:
                self._handle_movement(label, movement, now)
            else:
                self._log_dropped(label, tracker)

    def _log_dropped(self, label, tracker):
        """Logs an open-hand movement that was too short or small to count (helps tuning)."""
        dropped, tracker.last_movement = tracker.last_movement, None
        if dropped and tracker.last_problem and mv.open_fraction(dropped) >= cfg.MOVE_OPEN_SHARE:
            self._log(f"{label} hand movement ignored ({mv.describe(dropped)}): {tracker.last_problem}")

    def _handle_movement(self, label, movement, now):
        if cfg.MOVEMENT_NEEDS_OPEN_HAND and mv.open_fraction(movement) < cfg.MOVE_OPEN_SHARE:
            return   # moved while making a pose (pointing, a fist...) - not a movement gesture
        if now - self.last_movement_time.get(label, 0) < cfg.MOVE_REFRACTORY:
            self._log(f"{label} hand movement ignored ({mv.describe(movement)}): too soon after the previous one")
            return   # usually the hand coming back after a swipe
        name, distance = self.movements.recognize(movement)
        self._set_trail(label, movement, name, now)
        if name is not None:
            self.last_movement_time[label] = now   # ignore the hand coming back, whatever this one does
        if name is None:
            ranked = list(self.movements.distances(movement, try_shorter=True).items())
            if not ranked:
                reason = "no movements recorded"
            elif ranked[0][0] == mv.NONE_MOVEMENT:
                reason = "matched a corrected mistake"
            elif ranked[0][1] > self.movements.threshold:
                reason = f"closest '{ranked[0][0]}' {ranked[0][1]:.3f} > limit {self.movements.threshold:.3f}"
            else:
                second = next(((n, d) for n, d in ranked[1:] if n != mv.NONE_MOVEMENT), ("?", 0))
                reason = (f"unsure between '{ranked[0][0]}' {ranked[0][1]:.3f} and "
                          f"'{second[0]}' {second[1]:.3f}")
            self.last_action_label.setText("Last action: movement not recognised")
            self._log(f"{label} hand movement not recognised ({mv.describe(movement)}): {reason}")
            return
        action_key = self.gesture_mapping.get(name, "none")
        action = ACTIONS[action_key]
        if action_key == "none" or action.kind in ("cursor", "hold", "hold_move"):
            self.last_action_label.setText(f"Last action: {label} hand movement {name} (no action)")
            return
        if now - self.last_action_time.get(action_key, 0) < cfg.ACTION_COOLDOWN:
            return
        self.last_action_time[action_key] = now
        if action_key == "mark_mistake":
            self.mark_mistake(f"{name} movement")
            return
        try:
            action.fn()
            self.last_fired = {"time": now, "hand": label, "gesture": name, "action": action.label,
                               "kind": "movement", "path": movement}
            self.last_action_label.setText(f"Last action: {label} hand movement {name} -> {action.label}")
            self._log(f"{label} hand movement: {name} -> {action.label} (DTW distance {distance:.3f})")
        except Exception as e:
            self.last_action_label.setText(f"Action failed: {e}")
            self._log(f"{label} hand movement: {name} -> {action.label} FAILED: {e}")

    def _move_cursor(self, landmarks):
        self.cursor_used = True
        target = g.cursor_position(landmarks, self.cursor_sensitivity, SCREEN_W, SCREEN_H)
        position = self.cursor.update(*target, time.time())
        if position is not None:
            pyautogui.moveTo(*position)

    # ---------- Press-and-hold buttons ----------

    def _press_button(self, label, gesture, action):
        try:
            pyautogui.mouseDown(button=action.fn)
        except Exception as e:
            self._log(f"{label} hand: {gesture} -> {action.label} FAILED: {e}")
            return
        self.held_buttons[label] = {"button": action.fn, "gesture": gesture, "label": action.label,
                                    "kind": action.kind}
        self.last_action_label.setText(f"Last action: {label} hand {gesture} -> {action.label} (holding)")
        self._log(f"{label} hand: {gesture} -> {action.label} (pressed)")

    def _release_button(self, label):
        held = self.held_buttons.pop(label, None)
        if held is None:
            return
        try:
            pyautogui.mouseUp(button=held["button"])
        except Exception as e:
            self._log(f"Releasing the {held['button']} button failed: {e}")
        self.last_action_label.setText(f"Last action: {label} hand released {held['button']} button")
        self._log(f"{label} hand: {held['gesture']} -> {held['label']} (released)")

    def _release_all_buttons(self):
        for label in list(getattr(self, "held_buttons", {})):
            self._release_button(label)

    def _release_finished_holds(self):
        """Lets go of a button once its hand has stopped making the gesture for a few frames
        (a short flicker in recognition doesn't drop what you're dragging)."""
        for label, held in list(self.held_buttons.items()):
            if self.hand_states[label].latched != held["gesture"]:
                self._release_button(label)

    def _show_gestures_only(self, hands):
        """While paused: recognise and show each hand's gesture, but run no actions at all."""
        for label, landmarks in hands:
            gesture, confidence = g.predict(self.model, self.encoder, landmarks)
            if gesture in self.removed_poses:
                gesture = None
            self._set_hand_view(label, gesture if confidence >= self.confidence_threshold else None,
                                confidence, None)
        self.gesture_label.setText("Paused - gestures shown, no actions")
        self.confidence_label.setText(f"{len(hands)} hand{'s' if len(hands) != 1 else ''} in view")

    def _process_hand(self, label, landmarks):
        """Predicts one hand's gesture and runs its action. Returns (stable_gesture, confidence, action_key)."""
        features = g.normalize_landmarks(landmarks)
        gesture, confidence = g.predict_features(self.model, self.encoder, features)
        if gesture in self.removed_poses:
            gesture = None   # deleted, waiting for Retrain: ignore it
        state = self.hand_states[label]
        state.features.append(features)
        state.push(gesture if confidence >= self.confidence_threshold else None)
        stable = state.stable()
        held = self.held_buttons.get(label)
        if held is not None and held["kind"] == "hold_move":
            self._move_cursor(landmarks)   # dragging: follow the hand every frame until the button is released
            if stable is None:
                return None, confidence, None
        if stable is None:
            # Moving the cursor is harmless if briefly wrong, so it uses a looser check than clicks:
            # a single uncertain frame shouldn't make the cursor stop and stutter.
            loose = state.majority(cfg.CURSOR_AGREE_FRAMES)
            if loose is not None and ACTIONS[self.gesture_mapping.get(loose, "none")].kind == "cursor":
                self._move_cursor(landmarks)
                return loose, confidence, self.gesture_mapping[loose]
            return None, confidence, None

        action_key = self.gesture_mapping.get(stable, "none")
        action = ACTIONS[action_key]
        if action.kind == "cursor":
            self._move_cursor(landmarks)
        elif action.kind in ("hold", "hold_move"):
            if action.kind == "hold_move" and held is None:
                self._move_cursor(landmarks)   # (once held, the cursor is moved above, every frame)
            if label not in self.held_buttons:
                self._press_button(label, stable, action)
            state.latched, state.frames_released = stable, 0   # released once the gesture stops
        elif self.movement_trackers[label].active:
            pass   # in the middle of a movement: its frames aren't a held pose
        elif action_key != "none" and not (action.kind == "once" and state.latched == stable):
            cooldown = cfg.ACTION_COOLDOWN if action.kind == "once" else cfg.REPEAT_COOLDOWN
            now = time.time()
            if now - self.last_action_time.get(action_key, 0) >= cooldown:
                self.last_action_time[action_key] = now
                if action.kind == "once":
                    state.latched, state.frames_released = stable, 0   # don't repeat while held
                if action_key == "mark_mistake":
                    self.mark_mistake(f"{stable} gesture")
                    return stable, confidence, action_key
                try:
                    action.fn()
                    self.last_fired = {"time": now, "hand": label, "gesture": stable,
                                       "action": action.label, "frames": list(state.features)}
                    self.last_action_label.setText(f"Last action: {label} hand {stable} -> {action.label}")
                    self._log(f"{label} hand: {stable} -> {action.label}")
                except Exception as e:
                    self.last_action_label.setText(f"Action failed: {e}")
                    self._log(f"{label} hand: {stable} -> {action.label} FAILED: {e}")
        return stable, confidence, action_key

    # ---------- Two-hand zoom ----------

    def _apply_zoom(self, hand_a, hand_b):
        distance = g.two_hand_distance(hand_a, hand_b)
        previous, self.prev_hand_distance = self.prev_hand_distance, distance
        if previous is None or abs(distance - previous) < cfg.ZOOM_DEADZONE:
            return
        amount = int(min(abs(distance - previous) * cfg.ZOOM_SENSITIVITY, cfg.ZOOM_MAX_STEP))
        if amount < 1:
            return
        if not self.zoom_ctrl_held:
            pyautogui.keyDown("ctrl")
            self.zoom_ctrl_held = True
        zoom_in = distance > previous
        pyautogui.scroll(amount if zoom_in else -amount)
        self.last_action_label.setText("Last action: " + ("Hands apart -> Zoom in" if zoom_in
                                                          else "Hands together -> Zoom out"))

    def _reset_zoom(self):
        self.prev_hand_distance = None
        self._release_zoom_ctrl()

    def _release_zoom_ctrl(self):
        if self.zoom_ctrl_held:
            self.zoom_ctrl_held = False
            try:
                pyautogui.keyUp("ctrl")
            except Exception:
                pass

    def closeEvent(self, event):
        self.timer.stop()
        self._release_all_buttons()
        self._release_zoom_ctrl()
        self.camera.release()
        self.hands.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # draws the same on every Windows version, and follows the stylesheet closely
    palette = app.palette()   # dark defaults for anything the stylesheet doesn't cover (dialogs, scroll areas)
    for role, color in ((QPalette.Window, "#120c20"), (QPalette.Base, "#160f29"), (QPalette.AlternateBase, "#1b1430"),
                        (QPalette.WindowText, "#ece6fa"), (QPalette.Text, "#ece6fa"), (QPalette.Button, "#241a3d"),
                        (QPalette.ButtonText, "#ece6fa"), (QPalette.Highlight, "#7c4dff"),
                        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#241a3d"),
                        (QPalette.ToolTipText, "#ece6fa")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
    window = KinesisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
