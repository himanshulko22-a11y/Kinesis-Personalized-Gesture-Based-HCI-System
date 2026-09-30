"""
Kinesis - Personalized Gesture-Based HCI System

Control your computer with hand gestures using your webcam.
Record your own gestures, train a small neural network, and map
each gesture to an action (click, scroll, volume, screenshot, etc.).

Author: Himanshu Chauhan
License: MIT
Run: python kinesis_app.py  (install packages with requirements.txt)
"""
import sys
import os
import shutil
import json
import time
import pickle
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QScrollArea, QFrame, QMessageBox,
    QSlider, QCheckBox, QListWidget, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from sklearn.metrics import confusion_matrix

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.keras"
ENCODER_PATH = "label_encoder.pkl"
MAPPING_PATH = "gesture_mapping.json"

BACKUP_CSV_PATH = "gesture_data_backup.csv"
BACKUP_MODEL_PATH = "gesture_model_backup.keras"
BACKUP_ENCODER_PATH = "label_encoder_backup.pkl"
BACKUP_MAPPING_PATH = "gesture_mapping_backup.json"

CONFIDENCE_THRESHOLD = 0.70    # minimum model confidence before an action fires
SMOOTHING_WINDOW = 5            # frames in a row that must agree on a gesture
ACTION_COOLDOWN_SECONDS = 1.0    # wait before the same action can repeat
VOLUME_COOLDOWN_SECONDS = 0.3    
VOLUME_PRESSES_PER_TRIGGER = 3
FRAMES_PER_RECORDING = 150       # frames captured each time you record a gesture

ACTIVE_ZONE_X_MIN = 0.15
ACTIVE_ZONE_X_MAX = 0.85
ACTIVE_ZONE_Y_MIN = 0.15
ACTIVE_ZONE_Y_MAX = 0.65
EDGE_BUFFER = 5

ZOOM_DISTANCE_HISTORY_LEN = 8
ZOOM_DISPLACEMENT_THRESHOLD = 0.15
ZOOM_COOLDOWN_SECONDS = 1.0
ZOOM_SCROLL_AMOUNT = 300

ACTIVITY_LOG_MAX_ENTRIES = 50

IDLE_LOCK_TIMEOUT_SECONDS = 30    # lock the app after this long with no hand seen
IDLE_LOCK_WARNING_SECONDS = 10
UNLOCK_HOLD_SECONDS = 1.5

ACTIVITY_LOG_FILE_PATH = "activity_log.txt"
ACTIVITY_LOG_MAX_FILE_BYTES = 200_000

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False

DEFAULT_MAPPING = {
    "fist": "double_click",
    "open_palm": "none",
    "peace": "screenshot",
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
    "point": "move_cursor",
    "ok_sign": "right_click",
    "finger_gun": "double_click",
}

def press_volume_up():
    for _ in range(VOLUME_PRESSES_PER_TRIGGER):
        pyautogui.press("volumeup")

def press_volume_down():
    for _ in range(VOLUME_PRESSES_PER_TRIGGER):
        pyautogui.press("volumedown")

def press_scroll_up():
    pyautogui.scroll(300)

def press_scroll_down():
    pyautogui.scroll(-300)

def press_zoom_in():
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(ZOOM_SCROLL_AMOUNT)
    finally:
        pyautogui.keyUp("ctrl")

def press_zoom_out():
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(-ZOOM_SCROLL_AMOUNT)
    finally:
        pyautogui.keyUp("ctrl")

ACTION_DEFINITIONS = {
    "none":         {"label": "None (unhooked)", "continuous": False, "fast_cooldown": False, "fn": lambda: None},
    "left_click":   {"label": "Left Click",       "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.click(button="left")},
    "right_click":  {"label": "Right Click",      "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.click(button="right")},
    "double_click": {"label": "Double Click",     "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.doubleClick()},
    "volume_up":    {"label": "Volume Up",        "continuous": False, "fast_cooldown": True,  "fn": press_volume_up},
    "volume_down":  {"label": "Volume Down",      "continuous": False, "fast_cooldown": True,  "fn": press_volume_down},
    "screenshot":   {"label": "Screenshot",       "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.screenshot(f"screenshot_{int(time.time())}.png")},
    "move_cursor":  {"label": "Move Cursor",      "continuous": True,  "fast_cooldown": False, "fn": None},
    "switch_window_next": {"label": "Switch Window (Next)",     "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.hotkey("alt", "tab")},
    "switch_window_prev": {"label": "Switch Window (Previous)", "continuous": False, "fast_cooldown": False, "fn": lambda: pyautogui.hotkey("alt", "shift", "tab")},
    "scroll_up":    {"label": "Scroll Up",   "continuous": False, "fast_cooldown": True, "fn": press_scroll_up},
    "scroll_down":  {"label": "Scroll Down", "continuous": False, "fast_cooldown": True, "fn": press_scroll_down},
}

MOMENTARY_ACTION_KEYS = [key for key, info in ACTION_DEFINITIONS.items() if not info["continuous"]]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def normalize_landmarks(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[0]
    normalized = []
    for lm in landmarks:
        normalized.append(lm.x - wrist.x)
        normalized.append(lm.y - wrist.y)
        normalized.append(lm.z - wrist.z)
    return normalized

def remap_range(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

def get_palm_center(hand_landmarks):
    landmarks = hand_landmarks.landmark
    points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    avg_x = sum(p.x for p in points) / len(points)
    avg_y = sum(p.y for p in points) / len(points)
    return avg_x, avg_y

def get_two_hand_distance(hand_a, hand_b):
    ax, ay = get_palm_center(hand_a)
    bx, by = get_palm_center(hand_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def get_stable_gesture(recent_predictions):
    if len(recent_predictions) < SMOOTHING_WINDOW:
        return None
    if len(set(recent_predictions)) == 1:
        return recent_predictions[0]
    return None

def load_mapping(known_gestures):
    mapping = {}
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            mapping = json.load(f)

    for gesture, action in list(mapping.items()):
        if action not in ACTION_DEFINITIONS:
            mapping[gesture] = DEFAULT_MAPPING.get(gesture, "none")

    for gesture in known_gestures:
        if gesture not in mapping:
            mapping[gesture] = DEFAULT_MAPPING.get(gesture, "none")

    seen_actions = set()
    for gesture, action in mapping.items():
        if action == "none":
            continue
        if action in seen_actions:
            mapping[gesture] = "none"
        else:
            seen_actions.add(action)

    save_mapping(mapping)
    return mapping

def save_mapping(mapping):
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

def augment_landmarks(X, y, copies=2, noise_std=0.01, scale_range=0.05, seed=42):
    rng = np.random.default_rng(seed)
    all_X = [X]
    all_y = [y]
    for _ in range(copies):
        noise = rng.normal(0, noise_std, X.shape)
        scale = rng.uniform(1 - scale_range, 1 + scale_range, size=(X.shape[0], 1))
        all_X.append(X * scale + noise)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def train_model_from_csv(augment=True):
    keras.backend.clear_session()

    df = pd.read_csv(CSV_PATH)
    X = df.drop("label", axis=1).values
    y_text = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    raw_train_count = X_train.shape[0]
    if augment:
        X_train, y_train = augment_landmarks(X_train, y_train)
    augmented_train_count = X_train.shape[0]

    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=16, verbose=0)

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    return model, encoder, test_accuracy, raw_train_count, augmented_train_count

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #150f24;
    color: #e6def7;
    font-family: Segoe UI, Arial;
    font-size: 13px;
}
QLabel#appTitle {
    font-size: 26px;
    font-weight: bold;
    color: #b388ff;
    padding-bottom: 4px;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: bold;
    color: #c9a8ff;
    margin-top: 12px;
}
QLabel#gestureLabel {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
}
QLabel#confidenceLabel {
    font-size: 13px;
    color: #9d8ec2;
}
QLabel#lastActionLabel {
    font-size: 13px;
    color: #7ee787;
}
QFrame#videoFrame {
    background-color: #000000;
    border: 2px solid #6a3fc7;
    border-radius: 10px;
}
QPushButton {
    background-color: #6a3fc7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #8354e8;
}
QPushButton:pressed {
    background-color: #5a34a8;
}
QPushButton:disabled {
    background-color: #3a3350;
    color: #7a7290;
}
QComboBox {
    background-color: #241c3d;
    border: 1px solid #6a3fc7;
    border-radius: 6px;
    padding: 5px;
    color: #e6def7;
    min-width: 130px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #241c3d;
    color: #e6def7;
    selection-background-color: #6a3fc7;
}
QLineEdit {
    background-color: #241c3d;
    border: 1px solid #6a3fc7;
    border-radius: 6px;
    padding: 6px;
    color: #ffffff;
}
QScrollArea {
    border: none;
}
QPushButton#deleteBtn {
    background-color: #4a1f2e;
    color: #ff9eb0;
    padding: 6px 4px;
    font-size: 12px;
}
QPushButton#deleteBtn:hover {
    background-color: #6e2c42;
}
QSlider::groove:horizontal {
    background: #241c3d;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #b388ff;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
"""

class GestureMappingRow(QWidget):

    def __init__(self, gesture_name, current_action, on_change_callback,
                 on_record_more_callback, on_delete_callback):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)
        self.setLayout(layout)

        name_label = QLabel(gesture_name)
        name_label.setMinimumWidth(90)
        layout.addWidget(name_label)

        self.combo = QComboBox()
        for action_key, action_info in ACTION_DEFINITIONS.items():
            self.combo.addItem(action_info["label"], userData=action_key)

        index = self.combo.findData(current_action)
        if index >= 0:
            self.combo.setCurrentIndex(index)

        self.combo.currentIndexChanged.connect(
            lambda: on_change_callback(gesture_name, self.combo.currentData())
        )
        layout.addWidget(self.combo)

        record_more_btn = QPushButton("+")
        record_more_btn.setToolTip(f"Record more samples for '{gesture_name}'")
        record_more_btn.setFixedWidth(34)
        record_more_btn.clicked.connect(lambda: on_record_more_callback(gesture_name))
        layout.addWidget(record_more_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("deleteBtn")
        delete_btn.setFixedWidth(85)
        delete_btn.setStyleSheet("font-size: 11px; padding: 6px 4px;")
        delete_btn.clicked.connect(lambda: on_delete_callback(gesture_name))
        layout.addWidget(delete_btn)

class ConfusionMatrixDialog(QDialog):

    def __init__(self, class_names, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.resize(min(120 + 80 * len(class_names), 800), min(160 + 40 * len(class_names), 600))

        layout = QVBoxLayout()
        self.setLayout(layout)

        caption = QLabel("Rows = actual gesture, columns = predicted gesture. Diagonal = correct.")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #9d8ec2; font-size: 11px;")
        layout.addWidget(caption)

        n = len(class_names)
        table = QTableWidget(n, n)
        table.setHorizontalHeaderLabels(list(class_names))
        table.setVerticalHeaderLabels(list(class_names))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1

        for i in range(n):
            for j in range(n):
                value = int(matrix[i, j])
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                intensity = matrix[i, j] / row_max[i, 0]
                if i == j:
                    color = QColor(int(40 + (1 - intensity) * 20), int(60 + intensity * 120), int(50 + (1 - intensity) * 20))
                else:
                    color = QColor(int(60 + intensity * 150), int(40 + (1 - intensity) * 20), int(50 + (1 - intensity) * 20))
                item.setBackground(color)
                item.setForeground(QColor("#ffffff"))
                table.setItem(i, j, item)

        layout.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class KinesisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinesis - Personalized Gesture Control")
        self.setGeometry(80, 40, 1050, 780)

        self.setWindowIcon(QIcon(resource_path('favicon.ico')))

        print("Loading model...")
        self.model, self.encoder, self.gesture_mapping = self._load_model_safely()
        if self.encoder is not None:
            print(f"Model loaded. Gestures: {list(self.encoder.classes_)}")
        else:
            print("No trained model found yet. Record and train at least 2 gestures to begin.")

        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.recent_predictions_by_hand = {
            "Left": deque(maxlen=SMOOTHING_WINDOW),
            "Right": deque(maxlen=SMOOTHING_WINDOW),
        }
        self.last_action_time = {}
        self.consecutive_failed_reads = 0
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.cursor_sensitivity = 50
        self.paused = False

        self.augmentation_enabled = True

        self.zoom_enabled = False
        self.two_hand_distance_history = deque(maxlen=ZOOM_DISTANCE_HISTORY_LEN)
        self.last_zoom_time = 0

        self.app_mode = "normal"
        self.new_gesture_name = ""
        self.recording_buffer = []

        self.last_activity_time = time.time()
        self.unlock_hold_start = None
        self.unlock_gesture = None

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _load_model_safely(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
            return None, None, {}
        model = keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        mapping = load_mapping(list(encoder.classes_))
        return model, encoder, mapping

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QHBoxLayout()
        central.setLayout(outer_layout)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout()
        video_frame.setLayout(video_layout)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        outer_layout.addWidget(video_frame)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_scroll.setWidget(right_panel)

        title = QLabel("KINESIS")
        title.setObjectName("appTitle")
        right_layout.addWidget(title)

        self.gesture_label = QLabel("Gesture: none")
        self.gesture_label.setObjectName("gestureLabel")
        right_layout.addWidget(self.gesture_label)

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setObjectName("confidenceLabel")
        right_layout.addWidget(self.confidence_label)

        self.last_action_label = QLabel("Last action: -")
        self.last_action_label.setObjectName("lastActionLabel")
        right_layout.addWidget(self.last_action_label)

        unlock_gesture_row = QHBoxLayout()
        unlock_gesture_row.addWidget(QLabel("Unlock gesture:"))
        self.unlock_gesture_combo = QComboBox()
        self.unlock_gesture_combo.currentIndexChanged.connect(self.on_unlock_gesture_changed)
        unlock_gesture_row.addWidget(self.unlock_gesture_combo)
        right_layout.addLayout(unlock_gesture_row)
        self._refresh_unlock_gesture_options()

        pause_unlock_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause Recognition")
        self.pause_btn.setToolTip(
            "Freezes gesture -> action triggering. The camera preview and "
            "landmark tracking keep running, but nothing will click, scroll, "
            "or move the cursor until you resume."
        )
        self.pause_btn.clicked.connect(self.toggle_pause)
        pause_unlock_row.addWidget(self.pause_btn)

        self.unlock_btn = QPushButton("Manual Unlock")
        self.unlock_btn.setToolTip(
            f"Unlocks the app without using a gesture. Auto-lock triggers after "
            f"{IDLE_LOCK_TIMEOUT_SECONDS}s with no hand in frame; you can also "
            f"unlock by holding the selected gesture for {UNLOCK_HOLD_SECONDS}s."
        )
        self.unlock_btn.clicked.connect(lambda: self._unlock_app(method="manual button"))
        pause_unlock_row.addWidget(self.unlock_btn)
        right_layout.addLayout(pause_unlock_row)

        mapping_title = QLabel("Gesture Mapping")
        mapping_title.setObjectName("sectionTitle")
        right_layout.addWidget(mapping_title)

        self.mapping_layout = QVBoxLayout()
        self.mapping_layout.setAlignment(Qt.AlignTop)
        right_layout.addLayout(self.mapping_layout)

        self._refresh_mapping_rows()

        threshold_title = QLabel("Confidence Threshold")
        threshold_title.setObjectName("sectionTitle")
        right_layout.addWidget(threshold_title)

        threshold_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(95)
        self.threshold_slider.setValue(int(self.confidence_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_row.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel(f"{int(self.confidence_threshold * 100)}%")
        self.threshold_value_label.setFixedWidth(40)
        threshold_row.addWidget(self.threshold_value_label)
        right_layout.addLayout(threshold_row)

        sensitivity_title = QLabel("Cursor Sensitivity")
        sensitivity_title.setObjectName("sectionTitle")
        right_layout.addWidget(sensitivity_title)

        sensitivity_row = QHBoxLayout()
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(100)
        self.sensitivity_slider.setValue(self.cursor_sensitivity)
        self.sensitivity_slider.setToolTip(
            "Higher = less hand movement needed to reach screen edges (faster, less precise).\n"
            "Lower = more hand movement needed (slower, more precise)."
        )
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        sensitivity_row.addWidget(self.sensitivity_slider)

        self.sensitivity_value_label = QLabel(str(self.cursor_sensitivity))
        self.sensitivity_value_label.setFixedWidth(40)
        sensitivity_row.addWidget(self.sensitivity_value_label)
        right_layout.addLayout(sensitivity_row)

        advanced_title = QLabel("Advanced Gestures")
        advanced_title.setObjectName("sectionTitle")
        right_layout.addWidget(advanced_title)

        self.zoom_checkbox = QCheckBox("Two hands apart/together = Zoom in/out")
        self.zoom_checkbox.setToolTip(
            "Requires both hands visible AND both currently holding a "
            "gesture mapped to 'None (unhooked)' in your Gesture Mapping "
            "above (e.g. open_palm, by default) - this keeps zoom from "
            "firing while either hand is mid-gesture for a different "
            "mapped action. With both hands unhooked, moving them apart "
            "fast enough zooms in (Ctrl+scroll up); moving them together "
            "zooms out (Ctrl+scroll down). Works in Paint, browsers, "
            "PDFs, Office apps, and most image/code editors."
        )
        self.zoom_checkbox.setChecked(self.zoom_enabled)
        self.zoom_checkbox.stateChanged.connect(self.on_zoom_toggled)
        right_layout.addWidget(self.zoom_checkbox)

        advanced_note = QLabel(
            "Note: these work from raw hand position/motion, independent of "
            "your trained gestures above - so they work even mid-transition "
            "between poses. Gestures/actions above are recognized per-hand, "
            "so each hand can trigger a different mapped action at the same "
            "time (e.g. left hand moves the cursor while right hand clicks)."
        )
        advanced_note.setWordWrap(True)
        advanced_note.setStyleSheet("color: #9d8ec2; font-size: 11px;")
        right_layout.addWidget(advanced_note)

        open_folder_btn = QPushButton("Open Screenshots Folder")
        open_folder_btn.clicked.connect(self.open_screenshots_folder)
        right_layout.addWidget(open_folder_btn)

        log_title = QLabel("Activity Log")
        log_title.setObjectName("sectionTitle")
        right_layout.addWidget(log_title)

        self.activity_log_widget = QListWidget()
        self.activity_log_widget.setFixedHeight(140)
        right_layout.addWidget(self.activity_log_widget)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.activity_log_widget.clear)
        right_layout.addWidget(clear_log_btn)

        dataset_title = QLabel("Dataset Info")
        dataset_title.setObjectName("sectionTitle")
        right_layout.addWidget(dataset_title)

        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.setAlignment(Qt.AlignTop)
        right_layout.addLayout(self.dataset_layout)

        dataset_button_row = QHBoxLayout()
        refresh_dataset_btn = QPushButton("Refresh Info")
        refresh_dataset_btn.clicked.connect(self._refresh_dataset_info)
        dataset_button_row.addWidget(refresh_dataset_btn)

        balance_dataset_btn = QPushButton("Balance Dataset")
        balance_dataset_btn.setToolTip(
            "Trims every gesture down to match your smallest gesture's frame count, "
            "so no single gesture dominates training."
        )
        balance_dataset_btn.clicked.connect(self.balance_dataset)
        dataset_button_row.addWidget(balance_dataset_btn)
        right_layout.addLayout(dataset_button_row)

        self._refresh_dataset_info()

        add_title = QLabel("Add New Gesture")
        add_title.setObjectName("sectionTitle")
        right_layout.addWidget(add_title)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. fist")
        right_layout.addWidget(self.name_input)

        self.augmentation_checkbox = QCheckBox("Train with data augmentation")
        self.augmentation_checkbox.setChecked(self.augmentation_enabled)
        self.augmentation_checkbox.setToolTip(
            "When checked, each recorded frame gets a couple of lightly "
            "jittered copies (small position noise + slight scale change) "
            "added to the training set before Retrain Model runs - can "
            "help generalization. Uncheck to train on the raw recorded "
            "frames only."
        )
        self.augmentation_checkbox.stateChanged.connect(self.on_augmentation_toggled)
        right_layout.addWidget(self.augmentation_checkbox)

        button_row = QHBoxLayout()
        self.record_btn = QPushButton("Record Gesture")
        self.record_btn.clicked.connect(self.start_recording)
        button_row.addWidget(self.record_btn)

        self.train_btn = QPushButton("Retrain Model")
        self.train_btn.clicked.connect(self.retrain_model)
        button_row.addWidget(self.train_btn)
        right_layout.addLayout(button_row)

        self.self_test_btn = QPushButton("Self-Test Accuracy")
        self.self_test_btn.setToolTip(
            "Re-evaluates the currently loaded model against a fresh random "
            "slice of gesture_data.csv and shows accuracy overall and per gesture. "
            "Note: since this reuses the already-trained model, some rows in the "
            "test slice may have also been used during its original training, so "
            "treat this as a sanity check, not a guarantee - live testing is still "
            "the real test."
        )
        self.self_test_btn.clicked.connect(self.run_self_test)
        right_layout.addWidget(self.self_test_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        danger_title = QLabel("Danger Zone")
        danger_title.setObjectName("sectionTitle")
        danger_title.setStyleSheet("color: #ff9eb0;")
        right_layout.addWidget(danger_title)

        reset_btn = QPushButton("Reset All Data (delete everything)")
        reset_btn.setObjectName("deleteBtn")
        reset_btn.clicked.connect(self.reset_all_data)
        right_layout.addWidget(reset_btn)

        self.restore_btn = QPushButton("Restore Defaults")
        self.restore_btn.setToolTip(
            "Deletes the current gesture data, model, encoder, and mapping, "
            "then restores all of them from your saved default backup files "
            "- instant, no retraining needed."
        )
        self.restore_btn.clicked.connect(self.restore_defaults)
        right_layout.addWidget(self.restore_btn)

        right_layout.addStretch()
        outer_layout.addWidget(right_scroll)
        self._update_restore_button_state()

    def _refresh_dataset_info(self):
        while self.dataset_layout.count():
            item = self.dataset_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.deleteLater()

        if not os.path.exists(CSV_PATH):
            self.dataset_layout.addWidget(QLabel("No data recorded yet."))
            return

        df = pd.read_csv(CSV_PATH)
        counts = df["label"].value_counts()
        total = len(df)

        total_label = QLabel(f"Total frames: {total}")
        total_label.setStyleSheet("font-weight: bold; color: #c9a8ff;")
        self.dataset_layout.addWidget(total_label)

        average_count = total / len(counts) if len(counts) > 0 else 0

        for gesture_name, count in counts.items():
            row_text = f"{gesture_name}: {count} frames"
            if count < average_count * 0.5:
                row_text += "  (low - consider recording more)"
                color = "#ffb08a"
            else:
                color = "#e6def7"
            row_label = QLabel(row_text)
            row_label.setStyleSheet(f"color: {color};")
            self.dataset_layout.addWidget(row_label)

    def _refresh_mapping_rows(self):
        while self.mapping_layout.count():
            item = self.mapping_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.deleteLater()

        if self.encoder is None:
            self.mapping_layout.addWidget(QLabel("No gestures trained yet."))
            return

        for gesture_name in self.encoder.classes_:
            action = self.gesture_mapping.get(gesture_name, "none")
            row = GestureMappingRow(
                gesture_name, action,
                self.on_mapping_changed,
                self.record_more_samples,
                self.delete_gesture,
            )
            self.mapping_layout.addWidget(row)

    def on_mapping_changed(self, gesture_name, new_action_key):
        if new_action_key != "none":
            conflicting_gesture = next(
                (g for g, a in self.gesture_mapping.items()
                 if g != gesture_name and a == new_action_key),
                None,
            )
            if conflicting_gesture is not None:
                action_label = ACTION_DEFINITIONS[new_action_key]["label"]
                confirm = QMessageBox.question(
                    self, "Action already in use",
                    f"'{action_label}' is already hooked to '{conflicting_gesture}'.\n\n"
                    f"Move it to '{gesture_name}' instead? "
                    f"'{conflicting_gesture}' will be set to None.",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if confirm != QMessageBox.Yes:
                    self._refresh_mapping_rows()
                    return
                self.gesture_mapping[conflicting_gesture] = "none"

        self.gesture_mapping[gesture_name] = new_action_key
        save_mapping(self.gesture_mapping)
        self._refresh_mapping_rows()
        self._log_activity(f"Mapping updated: {gesture_name} -> {ACTION_DEFINITIONS[new_action_key]['label']}")

    def on_threshold_changed(self, value):
        self.confidence_threshold = value / 100.0
        self.threshold_value_label.setText(f"{value}%")

    def on_sensitivity_changed(self, value):
        self.cursor_sensitivity = value
        self.sensitivity_value_label.setText(str(value))

    def on_augmentation_toggled(self, state):
        self.augmentation_enabled = (state == Qt.Checked)

    def on_zoom_toggled(self, state):
        self.zoom_enabled = (state == Qt.Checked)
        self.two_hand_distance_history.clear()

    def _refresh_unlock_gesture_options(self):
        self.unlock_gesture_combo.blockSignals(True)
        self.unlock_gesture_combo.clear()

        if self.encoder is not None:
            for gesture_name in self.encoder.classes_:
                self.unlock_gesture_combo.addItem(gesture_name, userData=gesture_name)

        if self.unlock_gesture_combo.count() == 0:
            self.unlock_gesture_combo.addItem("(no gestures trained)", userData=None)
            self.unlock_gesture = None
        else:
            index = self.unlock_gesture_combo.findData(self.unlock_gesture)
            if index < 0:
                index = self.unlock_gesture_combo.findData("open_palm")
            if index < 0:
                index = 0
            self.unlock_gesture_combo.setCurrentIndex(index)
            self.unlock_gesture = self.unlock_gesture_combo.currentData()

        self.unlock_gesture_combo.blockSignals(False)

    def on_unlock_gesture_changed(self):
        self.unlock_gesture = self.unlock_gesture_combo.currentData()

    def _clear_prediction_state(self):
        for buffer in self.recent_predictions_by_hand.values():
            buffer.clear()
        self.two_hand_distance_history.clear()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("Resume Recognition" if self.paused else "Pause Recognition")
        self._clear_prediction_state()
        if self.paused:
            self.gesture_label.setText("Gesture: (paused)")
            self.confidence_label.setText("Confidence: -")
        self._log_activity("Recognition paused" if self.paused else "Recognition resumed")

    def _enter_locked_mode(self, reason="idle timeout"):
        if self.app_mode == "locked":
            return
        self.app_mode = "locked"
        self._clear_prediction_state()
        self.unlock_hold_start = None
        self.gesture_label.setText("Locked")
        self.confidence_label.setText("Confidence: -")
        self.status_label.setText(f"Locked - hold '{self.unlock_gesture}' or click Manual Unlock")
        self._log_activity(f"Auto-lock ({reason})")

    def _unlock_app(self, method="manual"):
        if self.app_mode != "locked":
            return
        self.app_mode = "normal"
        self.unlock_hold_start = None
        self.last_activity_time = time.time()
        self._clear_prediction_state()
        self.status_label.setText("")
        self._log_activity(f"Unlocked ({method})")

    def _check_unlock(self, hand_label, hand_landmarks):
        if self.unlock_gesture is None:
            return False

        features = np.array(normalize_landmarks(hand_landmarks)).reshape(1, -1)
        probabilities = self.model.predict(features, verbose=0)[0]
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]
        predicted_label = self.encoder.classes_[predicted_index]

        is_target = (predicted_label == self.unlock_gesture and confidence >= self.confidence_threshold)
        now = time.time()

        if not is_target:
            self.unlock_hold_start = None
            self.confidence_label.setText("Confidence: -")
            self.status_label.setText(f"Locked - hold '{self.unlock_gesture}' or click Manual Unlock")
            return False

        if self.unlock_hold_start is None:
            self.unlock_hold_start = now
        held_for = now - self.unlock_hold_start
        progress = min(100, int((held_for / UNLOCK_HOLD_SECONDS) * 100))
        self.confidence_label.setText(f"Confidence: {confidence*100:.0f}%")
        self.status_label.setText(f"Hold '{self.unlock_gesture}' to unlock: {progress}%")

        if held_for >= UNLOCK_HOLD_SECONDS:
            self._unlock_app(method=f"gesture ({self.unlock_gesture})")
            return True
        return False

    def _log_activity(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.activity_log_widget.insertItem(0, f"[{timestamp}] {message}")
        while self.activity_log_widget.count() > ACTIVITY_LOG_MAX_ENTRIES:
            self.activity_log_widget.takeItem(self.activity_log_widget.count() - 1)
        self._append_to_log_file(message)

    def _append_to_log_file(self, message):
        try:
            self._rotate_log_file_if_needed()
            full_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(ACTIVITY_LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(f"{full_timestamp} | {message}\n")
        except OSError as e:
            print(f"Activity log write failed: {e}")

    def _rotate_log_file_if_needed(self):
        try:
            if (os.path.exists(ACTIVITY_LOG_FILE_PATH)
                    and os.path.getsize(ACTIVITY_LOG_FILE_PATH) > ACTIVITY_LOG_MAX_FILE_BYTES):
                backup_path = ACTIVITY_LOG_FILE_PATH + ".1"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(ACTIVITY_LOG_FILE_PATH, backup_path)
        except OSError:
            pass

    def _show_video_overlay_message(self, message):
        self.video_label.clear()
        self.video_label.setText(message)
        self.video_label.setStyleSheet("color: #c9a8ff; font-size: 16px; font-weight: bold;")
        QApplication.processEvents()

    def open_screenshots_folder(self):
        folder = os.getcwd()
        try:
            os.startfile(folder)
        except Exception as e:
            QMessageBox.information(self, "Screenshots folder", f"Screenshots are saved in:\n{folder}")

    def start_recording(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Type a gesture name first.")
            return

        self.new_gesture_name = name
        self.recording_buffer = []
        self.app_mode = "recording"
        self.record_btn.setEnabled(False)
        self.status_label.setText(f"Recording '{name}'... hold your gesture steady.")

    def record_more_samples(self, gesture_name):
        self.new_gesture_name = gesture_name
        self.recording_buffer = []
        self.app_mode = "recording"
        self.record_btn.setEnabled(False)
        self.status_label.setText(f"Recording more samples for '{gesture_name}'... hold your gesture steady.")

    def balance_dataset(self):
        if not os.path.exists(CSV_PATH):
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return

        df = pd.read_csv(CSV_PATH)
        counts = df["label"].value_counts()

        if len(counts) < 2:
            QMessageBox.warning(self, "Not enough gestures", "Record at least 2 gestures first.")
            return

        min_count = counts.min()
        max_count = counts.max()

        if min_count == max_count:
            QMessageBox.information(self, "Already balanced", "All gestures already have the same frame count.")
            return

        confirm = QMessageBox.question(
            self, "Balance dataset?",
            f"This will trim every gesture down to {min_count} frames "
            f"(your smallest gesture's count). Frames beyond that are "
            f"permanently discarded. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        balanced_parts = []
        for label_name, group_df in df.groupby("label"):
            balanced_parts.append(group_df.sample(n=min_count, random_state=42))
        balanced_df = pd.concat(balanced_parts, ignore_index=True)
        balanced_df.to_csv(CSV_PATH, index=False)

        self.status_label.setText(f"Balanced dataset to {min_count} frames per gesture. Retraining...")
        self._show_video_overlay_message("Retraining model...")

        try:
            self.model, self.encoder, test_accuracy, _, _ = train_model_from_csv(augment=self.augmentation_enabled)
            self.gesture_mapping = load_mapping(list(self.encoder.classes_))
            self._refresh_mapping_rows()
            self._refresh_dataset_info()
            self._refresh_unlock_gesture_options()
            self.status_label.setText(f"Dataset balanced to {min_count} frames/gesture and retrained.")
            self._log_activity(f"Balanced dataset to {min_count} frames/gesture and retrained")
        except Exception as e:
            self.status_label.setText(f"Balanced data, but retraining failed: {e}")
            self._log_activity(f"Balance dataset retraining FAILED: {e}")

    def reset_all_data(self):
        confirm = QMessageBox.question(
            self, "Reset everything?",
            "This will PERMANENTLY delete:\n"
            "  - All recorded gesture data (gesture_data.csv)\n"
            "  - The trained model\n"
            "  - The label encoder\n"
            "  - Your gesture-to-action mapping\n\n"
            "Your saved default backup files are not affected - use "
            "'Restore Defaults' afterward if you want them back. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for path in (CSV_PATH, MODEL_PATH, ENCODER_PATH, MAPPING_PATH):
            if os.path.exists(path):
                os.remove(path)

        self.model = None
        self.encoder = None
        self.gesture_mapping = {}
        self._clear_prediction_state()
        self.last_action_time = {}

        self._refresh_mapping_rows()
        self._refresh_dataset_info()
        self._refresh_unlock_gesture_options()
        self._update_restore_button_state()
        self.gesture_label.setText("Gesture: none")
        self.confidence_label.setText("Confidence: -")
        self.last_action_label.setText("Last action: -")
        self.status_label.setText("Everything reset. Click 'Restore Defaults' to bring back your saved setup, or record fresh gestures.")
        self._log_activity("Full reset: all data, model, and mapping deleted")

    def restore_defaults(self):
        if not os.path.exists(BACKUP_MODEL_PATH) or not os.path.exists(BACKUP_ENCODER_PATH):
            QMessageBox.information(
                self, "No defaults available",
                "No default backup files were found "
                f"({BACKUP_MODEL_PATH}, {BACKUP_ENCODER_PATH}, etc.)."
            )
            return

        confirm = QMessageBox.question(
            self, "Restore defaults?",
            "This will delete your current gesture data, trained model, "
            "encoder, and mapping, and replace them with your saved "
            "default backup. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for path in (CSV_PATH, MODEL_PATH, ENCODER_PATH, MAPPING_PATH):
            if os.path.exists(path):
                os.remove(path)

        restore_pairs = [
            (BACKUP_CSV_PATH, CSV_PATH),
            (BACKUP_MODEL_PATH, MODEL_PATH),
            (BACKUP_ENCODER_PATH, ENCODER_PATH),
            (BACKUP_MAPPING_PATH, MAPPING_PATH),
        ]
        for backup, target in restore_pairs:
            if os.path.exists(backup):
                shutil.copy2(backup, target)

        try:
            self.model, self.encoder, self.gesture_mapping = self._load_model_safely()
            self._clear_prediction_state()
            self.last_action_time = {}
            self._refresh_mapping_rows()
            self._refresh_dataset_info()
            self._refresh_unlock_gesture_options()
            self._update_restore_button_state()
            self.status_label.setText(f"Restored defaults - {len(self.encoder.classes_)} gestures loaded, no retraining needed.")
            self._log_activity("Restored gestures/model from default backup")
        except Exception as e:
            self.status_label.setText(f"Restore failed: {e}")
            self._log_activity(f"Restore FAILED: {e}")

    def _update_restore_button_state(self):
        backup_exists = os.path.exists(BACKUP_MODEL_PATH) and os.path.exists(BACKUP_ENCODER_PATH)
        self.restore_btn.setEnabled(backup_exists)

    def delete_gesture(self, gesture_name):
        confirm = QMessageBox.question(
            self, "Delete gesture?",
            f"This will permanently delete all recorded data for '{gesture_name}' "
            f"and retrain the model without it. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        if not os.path.exists(CSV_PATH):
            return

        df = pd.read_csv(CSV_PATH)
        remaining_gestures = df["label"].unique()

        if len(remaining_gestures) <= 2:
            QMessageBox.warning(
                self, "Cannot delete",
                "You need at least 2 gestures remaining to train a classifier. "
                "Add another gesture before deleting this one."
            )
            return

        df = df[df["label"] != gesture_name]
        df.to_csv(CSV_PATH, index=False)

        if gesture_name in self.gesture_mapping:
            del self.gesture_mapping[gesture_name]
            save_mapping(self.gesture_mapping)

        self.status_label.setText(f"Deleted '{gesture_name}'. Retraining...")
        self._show_video_overlay_message("Retraining model...")

        try:
            self.model, self.encoder, test_accuracy, _, _ = train_model_from_csv(augment=self.augmentation_enabled)
            self.gesture_mapping = load_mapping(list(self.encoder.classes_))
            self._refresh_mapping_rows()
            self._refresh_dataset_info()
            self._refresh_unlock_gesture_options()
            self.status_label.setText(f"Deleted '{gesture_name}' and retrained on {len(self.encoder.classes_)} remaining gestures.")
            self._log_activity(f"Deleted gesture '{gesture_name}' and retrained")
        except Exception as e:
            self.status_label.setText(f"Deleted data, but retraining failed: {e}")
            self._log_activity(f"Deleted '{gesture_name}' but retraining FAILED: {e}")

    def finish_recording(self):
        if not os.path.exists(CSV_PATH):
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
            pd.DataFrame(columns=header).to_csv(CSV_PATH, index=False)

        rows = [[self.new_gesture_name] + frame for frame in self.recording_buffer]
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        new_df = pd.DataFrame(rows, columns=header)
        new_df.to_csv(CSV_PATH, mode="a", header=False, index=False)

        self.app_mode = "normal"
        self.record_btn.setEnabled(True)
        self._refresh_dataset_info()
        self.status_label.setText(
            f"Saved {len(self.recording_buffer)} frames for '{self.new_gesture_name}'. "
            f"Record a few more takes for variety, or click Retrain Model when ready."
        )
        self._log_activity(f"Recorded {len(self.recording_buffer)} frames for '{self.new_gesture_name}'")

    def retrain_model(self):
        if not os.path.exists(CSV_PATH):
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return

        self.status_label.setText("Training... this may take a few seconds.")
        self.train_btn.setEnabled(False)
        self._show_video_overlay_message("Retraining model...")

        try:
            self.model, self.encoder, test_accuracy, raw_count, augmented_count = train_model_from_csv(
                augment=self.augmentation_enabled
            )
            self.gesture_mapping = load_mapping(list(self.encoder.classes_))
            self._refresh_mapping_rows()
            self._refresh_dataset_info()
            self._refresh_unlock_gesture_options()
            if self.augmentation_enabled:
                frame_note = f"{raw_count} recorded frames -> {augmented_count} after augmentation"
            else:
                frame_note = f"{raw_count} recorded frames, augmentation off"
            self.status_label.setText(
                f"Retrained on {len(self.encoder.classes_)} gestures ({frame_note}). "
                f"(Note: test accuracy shown here is often inflated - trust live testing instead.)"
            )
            self._log_activity(f"Retrained model on {len(self.encoder.classes_)} gestures (train accuracy {test_accuracy*100:.1f}%, {frame_note})")
        except Exception as e:
            self.status_label.setText(f"Training failed: {e}")
            self._log_activity(f"Retraining FAILED: {e}")

        self.train_btn.setEnabled(True)

    def run_self_test(self):
        if self.model is None or self.encoder is None:
            QMessageBox.warning(self, "No model", "Train a model first (record at least 2 gestures, then Retrain Model).")
            return
        if not os.path.exists(CSV_PATH):
            QMessageBox.warning(self, "No data", "No gesture data recorded yet.")
            return

        df = pd.read_csv(CSV_PATH)
        X = df.drop("label", axis=1).values
        y_text = df["label"].values

        try:
            y = self.encoder.transform(y_text)
        except ValueError:
            QMessageBox.warning(
                self, "Model out of date",
                "The dataset has gestures the loaded model doesn't know about yet. "
                "Click Retrain Model first, then try Self-Test Accuracy again."
            )
            return

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None, stratify=y
        )

        probabilities = self.model.predict(X_test, verbose=0)
        predicted = np.argmax(probabilities, axis=1)
        overall_accuracy = float((predicted == y_test).mean())

        lines = [f"Overall accuracy: {overall_accuracy*100:.1f}%  ({len(y_test)} samples tested)", ""]
        for idx, class_name in enumerate(self.encoder.classes_):
            mask = y_test == idx
            if mask.sum() == 0:
                continue
            class_accuracy = float((predicted[mask] == idx).mean())
            lines.append(f"  {class_name}: {class_accuracy*100:.1f}%  ({int(mask.sum())} samples)")

        self._log_activity(f"Self-test run - overall accuracy {overall_accuracy*100:.1f}%")
        QMessageBox.information(self, "Self-Test Results", "\n".join(lines))

        cm = confusion_matrix(y_test, predicted, labels=range(len(self.encoder.classes_)))
        dialog = ConfusionMatrixDialog(self.encoder.classes_, cm, parent=self)
        dialog.exec_()

    def compute_active_zone(self):
        x_center = (ACTIVE_ZONE_X_MIN + ACTIVE_ZONE_X_MAX) / 2
        y_center = (ACTIVE_ZONE_Y_MIN + ACTIVE_ZONE_Y_MAX) / 2
        baseline_width = ACTIVE_ZONE_X_MAX - ACTIVE_ZONE_X_MIN
        baseline_height = ACTIVE_ZONE_Y_MAX - ACTIVE_ZONE_Y_MIN

        scale = remap_range(self.cursor_sensitivity, 1, 100, 1.6, 0.4)

        half_width = min(0.5, (baseline_width * scale) / 2)
        half_height = min(0.5, (baseline_height * scale) / 2)

        x_min = max(0.0, x_center - half_width)
        x_max = min(1.0, x_center + half_width)
        y_min = max(0.0, y_center - half_height)
        y_max = min(1.0, y_center + half_height)

        return x_min, x_max, y_min, y_max

    def move_cursor_from_landmark(self, hand_landmarks):
        palm_x, palm_y = get_palm_center(hand_landmarks)
        x_min, x_max, y_min, y_max = self.compute_active_zone()
        mapped_x = remap_range(palm_x, x_min, x_max, 0, 1)
        mapped_y = remap_range(palm_y, y_min, y_max, 0, 1)
        screen_x = int(mapped_x * SCREEN_WIDTH)
        screen_y = int(mapped_y * SCREEN_HEIGHT)
        screen_x = max(EDGE_BUFFER, min(SCREEN_WIDTH - EDGE_BUFFER, screen_x))
        screen_y = max(EDGE_BUFFER, min(SCREEN_HEIGHT - EDGE_BUFFER, screen_y))
        pyautogui.moveTo(screen_x, screen_y, duration=0)

    def _process_hand_gesture(self, hand_label, hand_landmarks):
        features = np.array(normalize_landmarks(hand_landmarks)).reshape(1, -1)
        probabilities = self.model.predict(features, verbose=0)[0]
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]
        predicted_label = self.encoder.classes_[predicted_index]

        buffer = self.recent_predictions_by_hand[hand_label]
        buffer.append(predicted_label if confidence >= self.confidence_threshold else None)
        stable_gesture = get_stable_gesture(buffer)

        is_continuous_action_active = False
        if stable_gesture is None:
            return None, confidence, is_continuous_action_active, None

        action_key = self.gesture_mapping.get(stable_gesture, "none")
        action_info = ACTION_DEFINITIONS[action_key]

        if action_info["continuous"]:
            is_continuous_action_active = True
            try:
                self.move_cursor_from_landmark(hand_landmarks)
            except Exception as e:
                print(f"Cursor move failed: {e}")
        elif action_key != "none":
            now = time.time()
            cooldown = VOLUME_COOLDOWN_SECONDS if action_info["fast_cooldown"] else ACTION_COOLDOWN_SECONDS
            last_time = self.last_action_time.get(action_key, 0)
            if now - last_time >= cooldown:
                try:
                    action_info["fn"]()
                    self.last_action_label.setText(f"Last action: {hand_label} hand {stable_gesture} -> {action_info['label']}")
                    self._log_activity(f"{hand_label} hand: {stable_gesture} -> {action_info['label']}")
                except Exception as e:
                    self.last_action_label.setText(f"Action failed: {e}")
                    self._log_activity(f"{hand_label} hand: {stable_gesture} -> {action_info['label']} FAILED: {e}")
                self.last_action_time[action_key] = now

        return stable_gesture, confidence, is_continuous_action_active, action_key

    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.consecutive_failed_reads += 1
            return
        self.consecutive_failed_reads = 0

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands_this_frame = []
        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for i, landmarks in enumerate(results.multi_hand_landmarks):
                label = (
                    handedness_list[i].classification[0].label
                    if i < len(handedness_list) else ("Left" if i == 0 else "Right")
                )
                hands_this_frame.append((label, landmarks))
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        primary_label, primary_hand = (hands_this_frame[0] if hands_this_frame else (None, None))

        if self.app_mode == "normal" and not self.paused and self.model is not None:
            if hands_this_frame:
                self.last_activity_time = time.time()
            else:
                idle_elapsed = time.time() - self.last_activity_time
                remaining = IDLE_LOCK_TIMEOUT_SECONDS - idle_elapsed
                if remaining <= 0:
                    self._enter_locked_mode(reason=f"idle {IDLE_LOCK_TIMEOUT_SECONDS}s")
                elif remaining <= IDLE_LOCK_WARNING_SECONDS:
                    self.status_label.setText(f"No hand detected - locking in {int(remaining) + 1}s...")
                elif self.status_label.text().startswith("No hand detected"):
                    self.status_label.setText("")

        if self.app_mode == "recording":
            if primary_hand is not None:
                self.recording_buffer.append(normalize_landmarks(primary_hand))
            self.status_label.setText(
                f"Recording '{self.new_gesture_name}': {len(self.recording_buffer)}/{FRAMES_PER_RECORDING}"
            )
            if len(self.recording_buffer) >= FRAMES_PER_RECORDING:
                self.finish_recording()

        elif self.paused:
            self.gesture_label.setText("Gesture: (paused)")
            self.confidence_label.setText("Confidence: -")

        elif self.app_mode == "normal" and hands_this_frame and self.model is not None:
            display_parts = []
            any_continuous_active = False
            primary_stable_gesture = None
            hand_action_keys = {}

            for label, landmarks in hands_this_frame:
                stable_gesture, confidence, is_continuous, action_key = self._process_hand_gesture(label, landmarks)
                if label == primary_label:
                    primary_stable_gesture = stable_gesture
                if is_continuous:
                    any_continuous_active = True
                hand_action_keys[label] = action_key
                display_parts.append(
                    f"{label}: {stable_gesture} ({confidence*100:.0f}%)" if stable_gesture is not None
                    else f"{label}: none"
                )

            seen_labels = {label for label, _ in hands_this_frame}
            for label, buffer in self.recent_predictions_by_hand.items():
                if label not in seen_labels:
                    buffer.append(None)

            self.gesture_label.setText("Gesture -> " + "  |  ".join(display_parts))
            self.confidence_label.setText(f"Hands detected: {len(hands_this_frame)}")

            if self.zoom_enabled:
                if len(hands_this_frame) == 2:
                    both_hands_unhooked = all(
                        hand_action_keys.get(label) == "none" for label, _ in hands_this_frame
                    )
                    if both_hands_unhooked:
                        self._check_zoom(hands_this_frame[0][1], hands_this_frame[1][1])
                    else:
                        self.two_hand_distance_history.clear()
                else:
                    self.two_hand_distance_history.clear()

        elif self.app_mode == "normal":
            self.gesture_label.setText("Gesture: none")
            self.confidence_label.setText("Confidence: -")
            for buffer in self.recent_predictions_by_hand.values():
                buffer.append(None)
            self.two_hand_distance_history.clear()

        elif self.app_mode == "locked":
            self.gesture_label.setText("Locked")
            if self.model is None:
                self.confidence_label.setText("Confidence: -")
                self.status_label.setText(
                    "Locked - no trained model loaded, so gesture unlock isn't "
                    "available. Click Manual Unlock, or Restore Defaults / "
                    "retrain first."
                )
            elif primary_hand is not None:
                self._check_unlock(primary_label, primary_hand)
            else:
                self.unlock_hold_start = None
                self.confidence_label.setText("Confidence: -")

        rgb_for_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_for_display.shape
        qt_image = QImage(rgb_for_display.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

    def _check_zoom(self, hand_a, hand_b):
        distance = get_two_hand_distance(hand_a, hand_b)
        self.two_hand_distance_history.append(distance)

        if len(self.two_hand_distance_history) < self.two_hand_distance_history.maxlen:
            return

        displacement = self.two_hand_distance_history[-1] - self.two_hand_distance_history[0]
        now = time.time()

        if now - self.last_zoom_time < ZOOM_COOLDOWN_SECONDS:
            return

        if displacement >= ZOOM_DISPLACEMENT_THRESHOLD:
            try:
                press_zoom_in()
                self.last_action_label.setText("Last action: Hands apart -> Zoom in")
                self._log_activity("Hands apart -> Zoom in")
            except Exception as e:
                print(f"Zoom in failed: {e}")
                self._log_activity(f"Zoom in FAILED: {e}")
            self.last_zoom_time = now
            self.two_hand_distance_history.clear()

        elif displacement <= -ZOOM_DISPLACEMENT_THRESHOLD:
            try:
                press_zoom_out()
                self.last_action_label.setText("Last action: Hands together -> Zoom out")
                self._log_activity("Hands together -> Zoom out")
            except Exception as e:
                print(f"Zoom out failed: {e}")
                self._log_activity(f"Zoom out FAILED: {e}")
            self.last_zoom_time = now
            self.two_hand_distance_history.clear()

    def closeEvent(self, event):
        self.timer.stop()
        self._show_video_overlay_message("Quitting...")
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = KinesisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
