"""Kinesis - personalized hand-gesture control for your computer.

Run:  python Kinesis_app.py

Files:
  config.py    settings and file paths
  actions.py   what a gesture can do (click, volume, scroll...)
  gestures.py  landmarks, dataset, model training, gesture -> action mapping
  widgets.py   small UI pieces (buttons, mapping row, confusion matrix)
  style.py     window theme
"""
import os
import shutil
import sys
import time

import cv2
import mediapipe as mp
import pyautogui
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

import config as cfg
import gestures as g
from actions import ACTIONS
from style import STYLESHEET
from widgets import ConfusionMatrixDialog, GestureMappingRow, ask, button, clear_layout, note, section_title

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
SCREEN_W, SCREEN_H = pyautogui.size()


# ---------- Main window ----------

class KinesisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinesis - Personalized Gesture Control")
        self.setGeometry(80, 40, 1050, 780)
        if os.path.exists(cfg.ICON_PATH):
            self.setWindowIcon(QIcon(cfg.ICON_PATH))

        load_error = self._load_model()

        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.CAMERA_INDEX, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.hand_states = {"Left": g.HandState(), "Right": g.HandState()}
        self.last_action_time = {}
        self.failed_reads = 0
        self.last_loop_error = None
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD
        self.cursor_sensitivity = 50
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

        self._build_ui()
        if load_error:
            self.status_label.setText(f"Saved model couldn't be loaded ({load_error}). Retrain or Restore Defaults.")
        elif not self.cap.isOpened():
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
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        QVBoxLayout(video_frame).addWidget(self.video_label)
        outer.addWidget(video_frame)

        panel = QWidget()
        col = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        outer.addWidget(scroll)

        title = QLabel("KINESIS")
        title.setObjectName("appTitle")
        col.addWidget(title)
        self.gesture_label = QLabel("Gesture: none")
        self.gesture_label.setObjectName("gestureLabel")
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setObjectName("confidenceLabel")
        self.last_action_label = QLabel("Last action: -")
        self.last_action_label.setObjectName("lastActionLabel")
        for label in (self.gesture_label, self.confidence_label, self.last_action_label):
            col.addWidget(label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Unlock gesture:"))
        self.unlock_combo = QComboBox()
        self.unlock_combo.currentIndexChanged.connect(
            lambda: setattr(self, "unlock_gesture", self.unlock_combo.currentData()))
        row.addWidget(self.unlock_combo)
        col.addLayout(row)

        row = QHBoxLayout()
        self.pause_btn = button("Pause Recognition", self.toggle_pause,
                                "Stops all actions. The camera keeps running.")
        row.addWidget(self.pause_btn)
        row.addWidget(button("Manual Unlock", lambda: self._unlock("manual button"),
                             f"The app locks after {cfg.IDLE_LOCK_TIMEOUT}s with no hand. Unlock here, "
                             f"or hold the unlock gesture for {cfg.UNLOCK_HOLD_SECONDS}s."))
        col.addLayout(row)

        col.addWidget(section_title("Gesture Mapping"))
        self.mapping_layout = QVBoxLayout()
        self.mapping_layout.setAlignment(Qt.AlignTop)
        col.addLayout(self.mapping_layout)

        self._add_slider(col, "Confidence Threshold", 50, 95, int(self.confidence_threshold * 100),
                         lambda v: f"{v}%", lambda v: setattr(self, "confidence_threshold", v / 100))
        self._add_slider(col, "Cursor Sensitivity", 1, 100, self.cursor_sensitivity,
                         str, lambda v: setattr(self, "cursor_sensitivity", v),
                         "Higher = less hand movement to reach the screen edges (faster, less precise).")

        col.addWidget(section_title("Advanced Gestures"))
        zoom_box = QCheckBox("Two hands apart/together = Zoom in/out")
        zoom_box.setToolTip("Works when both hands hold a gesture mapped to 'None' (e.g. open_palm). "
                            "Apart = Ctrl+scroll up, together = Ctrl+scroll down.")
        zoom_box.stateChanged.connect(self.on_zoom_toggled)
        col.addWidget(zoom_box)
        col.addWidget(note("Each hand is recognized separately, so the left hand can move the "
                           "cursor while the right hand clicks."))
        col.addWidget(button("Open Screenshots Folder", self.open_screenshots_folder))

        col.addWidget(section_title("Activity Log"))
        self.log_widget = QListWidget()
        self.log_widget.setFixedHeight(140)
        col.addWidget(self.log_widget)
        col.addWidget(button("Clear Log", self.log_widget.clear))

        col.addWidget(section_title("Dataset Info"))
        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.setAlignment(Qt.AlignTop)
        col.addLayout(self.dataset_layout)
        row = QHBoxLayout()
        row.addWidget(button("Refresh Info", self._refresh_dataset_info))
        row.addWidget(button("Balance Dataset", self.balance_dataset,
                             "Trims every gesture to the smallest gesture's frame count."))
        col.addLayout(row)

        col.addWidget(section_title("Add New Gesture"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. fist")
        col.addWidget(self.name_input)
        self.augment_box = QCheckBox("Train with data augmentation")
        self.augment_box.setChecked(True)
        self.augment_box.setToolTip("Adds jittered and mirrored copies of each frame when training.")
        col.addWidget(self.augment_box)

        row = QHBoxLayout()
        self.record_btn = button("Record Gesture", self.start_recording)
        tips_btn = button("ⓘ", self.show_recording_tips, "Tips for recording a gesture")
        tips_btn.setFixedWidth(28)
        self.save_btn = button("Save Gesture", self.save_recording)
        self.train_btn = button("Retrain Model", lambda: self._retrain("Retrained."))
        for widget in (self.record_btn, tips_btn, self.save_btn, self.train_btn):
            row.addWidget(widget)
        col.addLayout(row)
        col.addWidget(button("Self-Test Accuracy", self.run_self_test,
                             "Tests the model on a random slice of your data. Some of it was used in "
                             "training, so treat it as a sanity check - live testing is the real test."))

        row = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.retake_btn = button("Retake", lambda: self._begin_recording(self.new_gesture_name))
        self.cancel_btn = button("Cancel", self.cancel_recording, name="deleteBtn")
        for widget in (self.status_label, self.retake_btn, self.cancel_btn):
            row.addWidget(widget)
        col.addLayout(row)

        danger = section_title("Danger Zone")
        danger.setStyleSheet("color: #ff9eb0;")
        col.addWidget(danger)
        col.addWidget(button("Reset All Data (delete everything)", self.reset_all_data, name="deleteBtn"))
        row = QHBoxLayout()
        row.addWidget(button("Save as Default", self.save_as_default,
                             "Saves your current gestures, model and mapping as the default backup."))
        self.restore_btn = button("Restore Defaults", self.restore_defaults,
                                  "Replaces your current data with the saved default backup.")
        row.addWidget(self.restore_btn)
        col.addLayout(row)
        col.addStretch()

        self._set_recording_buttons()
        self._refresh_after_model_change()

    def _add_slider(self, layout, title, low, high, value, fmt, on_change, tooltip=None):
        layout.addWidget(section_title(title))
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

    def _refresh_after_model_change(self):
        self.gesture_mapping = g.load_mapping(list(self.encoder.classes_)) if self.encoder is not None else {}
        self._clear_prediction_state()
        self.last_action_time = {}
        self._refresh_mapping_rows()
        self._refresh_dataset_info()
        self._refresh_unlock_options()
        self.restore_btn.setEnabled(self._has_backup())

    def _refresh_dataset_info(self):
        clear_layout(self.dataset_layout)
        df = g.read_dataset()
        if df is None or df.empty:
            self.dataset_layout.addWidget(QLabel("No data recorded yet."))
            return
        counts = df["label"].value_counts()
        total = QLabel(f"Total frames: {len(df)}")
        total.setStyleSheet("font-weight: bold; color: #c9a8ff;")
        self.dataset_layout.addWidget(total)
        average = len(df) / len(counts)
        for gesture, count in counts.items():
            low = count < average * 0.5
            label = QLabel(f"{gesture}: {count} frames" + ("  (low - consider recording more)" if low else ""))
            label.setStyleSheet(f"color: {'#ffb08a' if low else '#e6def7'};")
            self.dataset_layout.addWidget(label)

    def _refresh_mapping_rows(self):
        clear_layout(self.mapping_layout)
        if self.encoder is None:
            self.mapping_layout.addWidget(QLabel("No gestures trained yet."))
            return
        for gesture in self.encoder.classes_:
            self.mapping_layout.addWidget(GestureMappingRow(
                gesture, self.gesture_mapping.get(gesture, "none"),
                self.on_mapping_changed, self._begin_recording, self.delete_gesture))

    def _refresh_unlock_options(self):
        combo = self.unlock_combo
        combo.blockSignals(True)
        combo.clear()
        for gesture in (self.encoder.classes_ if self.encoder is not None else []):
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
        for state in self.hand_states.values():
            state.reset()
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
        self.confidence_label.setText("Confidence: -")
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
        self.log_widget.insertItem(0, f"[{time.strftime('%H:%M:%S')}] {message}")
        while self.log_widget.count() > cfg.LOG_MAX_ENTRIES:
            self.log_widget.takeItem(self.log_widget.count() - 1)
        try:
            if os.path.exists(cfg.LOG_PATH) and os.path.getsize(cfg.LOG_PATH) > cfg.LOG_MAX_FILE_BYTES:
                os.replace(cfg.LOG_PATH, cfg.LOG_PATH + ".1")
            with open(cfg.LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
        except OSError as e:
            print(f"Activity log write failed: {e}")

    def _show_video_message(self, message):
        self.video_label.clear()
        self.video_label.setText(message)
        self.video_label.setStyleSheet("color: #c9a8ff; font-size: 16px; font-weight: bold;")
        QApplication.processEvents()

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
            "This variety helps the model recognize the gesture wherever your hand is later.")

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
        self._begin_recording(name)

    def _begin_recording(self, name):
        if self.mode in ("recording", "reviewing") and name != self.new_gesture_name:
            return   # finish or cancel the current recording first
        self.new_gesture_name = name
        self.recording_buffer = []
        self._clear_prediction_state()   # also releases a held Ctrl from zoom
        self.mode = "recording"
        self._set_recording_buttons()
        self.status_label.setText(f"Recording '{name}'... hold your gesture with one hand.")

    def _record_frame(self, hands):
        if len(hands) == 1:
            self.recording_buffer.append(g.normalize_landmarks(hands[0][1]))
            self.status_label.setText(
                f"Recording '{self.new_gesture_name}': {len(self.recording_buffer)}/{cfg.FRAMES_PER_RECORDING}")
        elif len(hands) > 1:
            self.status_label.setText("Show only ONE hand while recording.")
        if len(self.recording_buffer) >= cfg.FRAMES_PER_RECORDING:
            self.mode = "reviewing"
            self._set_recording_buttons()
            self.status_label.setText(
                f"Captured {len(self.recording_buffer)} frames for '{self.new_gesture_name}'. "
                f"Save Gesture to keep it, or Retake.")

    def save_recording(self):
        g.append_recording(self.new_gesture_name, self.recording_buffer)
        count = len(self.recording_buffer)
        self.recording_buffer = []
        self.mode = "normal"
        self._set_recording_buttons()
        self._refresh_dataset_info()
        self.status_label.setText(f"Saved {count} frames for '{self.new_gesture_name}'. "
                                  f"Record more takes for variety, or Retrain Model when ready.")
        self._log(f"Recorded {count} frames for '{self.new_gesture_name}'")

    def cancel_recording(self):
        self.recording_buffer = []
        self.mode = "normal"
        self.last_activity_time = time.time()
        self._set_recording_buttons()
        self.status_label.setText("Recording cancelled.")

    # ---------- Training and data management ----------

    def _retrain(self, done_message):
        if g.read_dataset() is None:
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return
        self.status_label.setText("Training... this may take a few seconds.")
        self.train_btn.setEnabled(False)
        self._show_video_message("Retraining model...")
        try:
            self.model, self.encoder, accuracy, raw, used = g.train_model(self.augment_box.isChecked())
        except Exception as e:
            self.status_label.setText(f"Training failed: {e}")
            self._log(f"Retraining FAILED: {e}")
            return
        finally:
            self.train_btn.setEnabled(True)
        self._refresh_after_model_change()
        frames = f"{raw} frames -> {used} after augmentation" if used != raw else f"{raw} frames"
        self.status_label.setText(f"{done_message} {len(self.encoder.classes_)} gestures, {frames}, "
                                  f"test accuracy {accuracy:.0%} (often optimistic - trust live testing).")
        self._log(f"{done_message} test accuracy {accuracy:.1%}, {frames}")

    def balance_dataset(self):
        df = g.read_dataset()
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
        df.groupby("label").sample(n=smallest, random_state=42).to_csv(cfg.CSV_PATH, index=False)
        self._retrain(f"Balanced to {smallest} frames per gesture and retrained.")

    def delete_gesture(self, gesture):
        df = g.read_dataset()
        if df is None:
            return
        if df["label"].nunique() <= 2:
            QMessageBox.warning(self, "Cannot delete", "You need at least 2 gestures to train. "
                                                       "Add another gesture before deleting this one.")
            return
        if not ask(self, "Delete gesture?", f"Permanently delete all data for '{gesture}' and retrain?"):
            return
        df[df["label"] != gesture].to_csv(cfg.CSV_PATH, index=False)
        self._retrain(f"Deleted '{gesture}' and retrained.")

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
        try:
            y = self.encoder.transform(df["label"].values)
            _, X_test, _, y_test = train_test_split(df.drop("label", axis=1).values, y,
                                                    test_size=0.2, stratify=y)
        except ValueError:
            QMessageBox.warning(self, "Model out of date",
                                "Your data has changed since the last training. Click Retrain Model first.")
            return

        predicted = self.model.predict(X_test, verbose=0).argmax(axis=1)
        lines = [f"Overall accuracy: {(predicted == y_test).mean():.1%}  ({len(y_test)} samples)", ""]
        for index, name in enumerate(self.encoder.classes_):
            mask = y_test == index
            if mask.any():
                lines.append(f"  {name}: {(predicted[mask] == index).mean():.1%}  ({mask.sum()} samples)")
        self._log(f"Self-test - overall accuracy {(predicted == y_test).mean():.1%}")
        QMessageBox.information(self, "Self-Test Results", "\n".join(lines))
        matrix = confusion_matrix(y_test, predicted, labels=range(len(self.encoder.classes_)))
        ConfusionMatrixDialog(self.encoder.classes_, matrix, parent=self).exec_()

    def _remove_data_files(self):
        for path in cfg.DATA_FILES:
            if os.path.exists(path):
                os.remove(path)

    def reset_all_data(self):
        if not ask(self, "Reset everything?",
                   "Permanently delete all recorded gestures, the trained model and your mapping?\n\n"
                   "Your saved defaults are kept - use Restore Defaults to get them back."):
            return
        self._remove_data_files()
        self.model = self.encoder = None
        self.mode = "normal"
        self.recording_buffer = []
        self._set_recording_buttons()
        self._refresh_after_model_change()
        self.gesture_label.setText("Gesture: none")
        self.confidence_label.setText("Confidence: -")
        self.last_action_label.setText("Last action: -")
        self.status_label.setText("Everything reset. Record new gestures, or Restore Defaults.")
        self._log("Full reset: all data, model, and mapping deleted")

    def save_as_default(self):
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
            if str(e) != self.last_loop_error:
                self.last_loop_error = str(e)
                self._log(f"Error: {e}")

    def _process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.failed_reads += 1
            if self.failed_reads == 30:
                self.status_label.setText("Camera not returning frames - check it's connected and not used by another app.")
            return
        self.failed_reads = 0

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = self._find_hands(self.hands.process(rgb), frame)

        now = time.time()
        if now - self.last_frame_time > 2.0 or self.paused or self.mode in ("recording", "reviewing"):
            self.last_activity_time = now   # don't count pauses, training or recording as idle time
        self.last_frame_time = now

        if self.mode == "recording":
            self._record_frame(hands)
        elif self.mode == "locked":
            self._update_locked(hands)
        elif self.mode == "normal":
            self._update_normal(hands, now)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

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
            self.confidence_label.setText("Confidence: -")
            self.status_label.setText(self._lock_message())
            return
        matches = [conf for gesture, conf in (g.predict(self.model, self.encoder, lm) for _, lm in hands)
                   if gesture == self.unlock_gesture and conf >= self.confidence_threshold]
        if not matches:
            self.unlock_hold_start = None
            self.confidence_label.setText("Confidence: -")
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
        if self.paused or QApplication.activeModalWidget() is not None:
            # Paused, or a dialog is open (a gesture click must never press "Yes" on Reset All Data).
            self._clear_prediction_state()
            self.gesture_label.setText("Gesture: (paused)" if self.paused else "Gesture: (dialog open)")
            self.confidence_label.setText("Confidence: -")
            return
        if self.model is None:
            self.gesture_label.setText("Gesture: none (record and train gestures first)")
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
            self.gesture_label.setText("Gesture: none")
            self.confidence_label.setText("Confidence: -")

        action_keys, parts = {}, []
        for label, landmarks in hands:
            gesture, confidence, action_key = self._process_hand(label, landmarks)
            action_keys[label] = action_key
            parts.append(f"{label}: {gesture} ({confidence:.0%})" if gesture else f"{label}: none")
        for label, state in self.hand_states.items():
            if label not in action_keys:
                state.push(None)
        if hands:
            self.gesture_label.setText("Gesture -> " + "  |  ".join(parts))
            self.confidence_label.setText(f"Hands detected: {len(hands)}")

        both_unhooked = len(hands) == 2 and all(action_keys.get(label) == "none" for label, _ in hands)
        if self.zoom_enabled and both_unhooked:
            self._apply_zoom(hands[0][1], hands[1][1])
        else:
            self._reset_zoom()

    def _process_hand(self, label, landmarks):
        """Predicts one hand's gesture and runs its action. Returns (stable_gesture, confidence, action_key)."""
        gesture, confidence = g.predict(self.model, self.encoder, landmarks)
        state = self.hand_states[label]
        state.push(gesture if confidence >= self.confidence_threshold else None)
        stable = state.stable()
        if stable is None:
            return None, confidence, None

        action_key = self.gesture_mapping.get(stable, "none")
        action = ACTIONS[action_key]
        if action.kind == "cursor":
            pyautogui.moveTo(*g.cursor_position(landmarks, self.cursor_sensitivity, SCREEN_W, SCREEN_H))
        elif action_key != "none" and not (action.kind == "once" and state.latched == stable):
            cooldown = cfg.ACTION_COOLDOWN if action.kind == "once" else cfg.REPEAT_COOLDOWN
            now = time.time()
            if now - self.last_action_time.get(action_key, 0) >= cooldown:
                self.last_action_time[action_key] = now
                if action.kind == "once":
                    state.latched, state.frames_released = stable, 0   # don't repeat while held
                try:
                    action.fn()
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
        self._release_zoom_ctrl()
        self.cap.release()
        self.hands.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = KinesisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
