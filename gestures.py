"""Hand landmarks, the gesture dataset, the model, and the gesture -> action mapping.

Nothing here touches the UI, so it can be tested on its own.
"""
import json
import math
import os
import pickle
from collections import deque

import numpy as np
import pandas as pd

import config as cfg
from actions import ACTIONS

NONE_LABEL = "_not_a_gesture"   # frames from corrected mistakes: "this is none of my gestures"
CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


# ---------- Landmarks ----------

def normalize_landmarks(hand):
    """63 numbers (x, y, z per landmark) relative to the wrist, so hand position doesn't matter."""
    wrist = hand.landmark[0]
    features = []
    for lm in hand.landmark:
        features += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
    return features


def palm_center(hand):
    points = [hand.landmark[i] for i in (0, 5, 9, 13, 17)]
    return sum(p.x for p in points) / 5, sum(p.y for p in points) / 5


def two_hand_distance(hand_a, hand_b):
    return math.dist(palm_center(hand_a), palm_center(hand_b))


def remap_range(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def cursor_position(hand, sensitivity, screen_w, screen_h):
    """Screen pixel for the cursor. Follows landmark 9 (base of the middle finger), which
    shakes less than the wrist or a fingertip. Higher sensitivity = smaller active zone."""
    point = hand.landmark[9]
    scale = remap_range(sensitivity, 1, 100, 1.6, 0.4)
    position = []
    for value, (low, high), size in ((point.x, cfg.ACTIVE_ZONE_X, screen_w),
                                     (point.y, cfg.ACTIVE_ZONE_Y, screen_h)):
        center, half = (low + high) / 2, min(0.5, (high - low) * scale / 2)
        fraction = remap_range(value, max(0.0, center - half), min(1.0, center + half), 0, 1)
        pixel = int(fraction * size)
        position.append(max(cfg.EDGE_BUFFER, min(size - cfg.EDGE_BUFFER, pixel)))
    return tuple(position)


# ---------- Smoothing ----------

class OneEuroFilter:
    """Smooths a noisy value: strong smoothing when it changes slowly (removes shake),
    light smoothing when it changes fast (stays responsive).
    Casiez, Roussel & Vogel, "1 Euro Filter", CHI 2012."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.value = None
        self.speed = 0.0
        self.time = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, value, t):
        if self.value is None:
            self.value, self.time = value, t
            return value
        dt = max(t - self.time, 1e-3)
        raw_speed = (value - self.value) / dt
        self.speed += self._alpha(self.d_cutoff, dt) * (raw_speed - self.speed)
        cutoff = self.min_cutoff + self.beta * abs(self.speed)
        self.value += self._alpha(cutoff, dt) * (value - self.value)
        self.time = t
        return self.value


class CursorSmoother:
    """Removes hand shake from the cursor. A One Euro filter smooths the position, and a
    hold zone keeps a resting cursor perfectly still until the hand clearly moves."""

    def __init__(self, steadiness=cfg.CURSOR_STEADINESS):
        self.filters = [OneEuroFilter(cfg.CURSOR_MIN_CUTOFF, cfg.CURSOR_BETA) for _ in range(2)]
        self.set_steadiness(steadiness)
        self.reset()

    def set_steadiness(self, value):
        self.hold_radius = remap_range(value, 0, 100, *cfg.CURSOR_HOLD_RADIUS)

    def reset(self):
        for f in self.filters:
            f.reset()
        self.position = None
        self.moving = False
        self.slow_frames = 0

    def update(self, x, y, t):
        """Raw target pixel in -> pixel to move the cursor to, or None to leave it where it is."""
        smoothed = tuple(f(v, t) for f, v in zip(self.filters, (x, y)))
        if self.position is not None:
            if self.moving:
                speed = math.hypot(self.filters[0].speed, self.filters[1].speed)
                self.slow_frames = self.slow_frames + 1 if speed < cfg.CURSOR_STILL_SPEED else 0
                self.moving = self.slow_frames < 3
            elif math.dist(smoothed, self.position) > self.hold_radius:
                self.moving, self.slow_frames = True, 0
            else:
                return None
        self.position = smoothed
        return round(smoothed[0]), round(smoothed[1])


class HandState:
    """Recent predictions for one hand, plus which one-shot gesture is still being held."""

    def __init__(self):
        self.recent = deque(maxlen=cfg.SMOOTHING_WINDOW)
        self.latched = None        # one-shot gesture that already fired and hasn't been released
        self.frames_released = 0
        self.features = deque(maxlen=cfg.MISTAKE_FRAMES)   # recent hand frames, saved if a mistake is marked

    def reset(self):
        self.recent.clear()
        self.features.clear()
        self.latched = None
        self.frames_released = 0

    def push(self, gesture):
        """gesture = confident prediction this frame, or None."""
        self.recent.append(gesture)
        if self.latched is not None:
            self.frames_released = 0 if gesture == self.latched else self.frames_released + 1
            if self.frames_released >= cfg.RELEASE_FRAMES:
                self.latched = None

    def stable(self):
        """The gesture if every recent frame agrees on it, else None."""
        if len(self.recent) == self.recent.maxlen and len(set(self.recent)) == 1:
            return self.recent[0]
        return None


# ---------- Similar-gesture warning ----------

def hand_shape(frames):
    """Hand shape only: x/y scaled to one hand size, so distance from the camera and depth don't count."""
    points = np.asarray(frames, dtype=np.float32).reshape(-1, 21, 3)[:, :, :2]
    size = np.linalg.norm(points[:, 9], axis=1) + 1e-9   # wrist -> base of middle finger
    return (points / size[:, None, None]).reshape(len(points), -1)


def _median_nearest(A, B):
    """Median, over frames in A, of the distance to the closest frame in B."""
    squared = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return float(np.median(np.sqrt(np.maximum(squared.min(axis=1), 0))))


def gesture_similarity(new_frames, df):
    """Hand-shape distance from a new recording to each recorded gesture (smaller = more alike).
    A mirrored hand counts as the same shape, as in training."""
    new = np.asarray(new_frames, dtype=np.float32)
    mirrored = new.copy()
    mirrored[:, 0::3] *= -1
    shapes = (hand_shape(new), hand_shape(mirrored))
    distances = {}
    for gesture, group in df[df["label"] != NONE_LABEL].groupby("label"):
        reference = hand_shape(group.drop(columns="label").values)
        distances[gesture] = min(_median_nearest(s, reference) for s in shapes)
    return dict(sorted(distances.items(), key=lambda item: item[1]))


def similarity_warning(name, new_frames):
    """Returns (level, message) if a new recording is too much like another gesture
    (or unlike its own earlier recordings), else None. level is "strong" or "mild"."""
    df = read_dataset()
    if df is None or not len(new_frames):
        return None
    distances = gesture_similarity(new_frames, df)
    own = distances.pop(name, None)
    for other, distance in distances.items():   # closest first
        if distance < cfg.SIMILAR_SAME:
            return "strong", (f"This looks almost the same as '{other}' (difference {distance:.2f}). "
                              f"The model will mix them up - use a more different hand shape, or Retake.")
        if distance < cfg.SIMILAR_CLOSE:
            return "mild", (f"This is quite similar to '{other}' (difference {distance:.2f}). "
                            f"It may sometimes be confused - check with Self-Test after training.")
        break
    if own is not None and own > cfg.SIMILAR_CLOSE:
        return "mild", (f"This take looks different from your earlier '{name}' recordings "
                        f"(difference {own:.2f}). Is it really the same gesture?")
    return None


# ---------- Dataset ----------

def read_dataset():
    return pd.read_csv(cfg.CSV_PATH) if os.path.exists(cfg.CSV_PATH) else None


def append_recording(gesture_name, frames):
    new_file = not os.path.exists(cfg.CSV_PATH)
    rows = [[gesture_name] + frame for frame in frames]
    pd.DataFrame(rows, columns=CSV_HEADER).to_csv(cfg.CSV_PATH, mode="a", header=new_file, index=False)


def augment_landmarks(X, y, copies=2, noise_std=0.01, scale_range=0.05, seed=42, mirror=True):
    """Adds jittered copies (noise + slight scale) and a mirrored copy, so a gesture
    recorded with one hand is also recognized on the other."""
    rng = np.random.default_rng(seed)
    all_X, all_y = [X], [y]
    for _ in range(copies):
        scale = rng.uniform(1 - scale_range, 1 + scale_range, size=(X.shape[0], 1))
        all_X.append(X * scale + rng.normal(0, noise_std, X.shape))
        all_y.append(y)
    if mirror:
        mirrored = X.copy()
        mirrored[:, 0::3] *= -1   # flip every x coordinate
        all_X.append(mirrored)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


# ---------- Model ----------

def load_trained_model():
    """Returns (model, encoder), or (None, None) if nothing is trained yet. Raises if the files are broken."""
    if not (os.path.exists(cfg.MODEL_PATH) and os.path.exists(cfg.ENCODER_PATH)):
        return None, None
    from tensorflow import keras
    model = keras.models.load_model(cfg.MODEL_PATH)
    with open(cfg.ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def train_model(augment=True):
    """Trains on the CSV and saves the model. Returns (model, encoder, held_out_accuracy, raw_rows, train_rows)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow import keras
    from tensorflow.keras import layers

    df = read_dataset()
    if df is not None and (df["label"] == NONE_LABEL).sum() < cfg.MIN_MISTAKE_FRAMES:
        df = df[df["label"] != NONE_LABEL]   # too few saved mistakes to learn from yet
    real = df[df["label"] != NONE_LABEL] if df is not None else None
    if real is None or real["label"].nunique() < 2:
        raise ValueError("Record at least 2 different gestures before training.")
    counts = real["label"].value_counts()
    if counts.min() < 10:
        raise ValueError(f"'{counts.idxmin()}' has only {counts.min()} frames - record it again.")

    X = df.drop("label", axis=1).values.astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    raw_rows = len(X_train)
    if augment:
        X_train, y_train = augment_landmarks(X_train, y_train)

    keras.backend.clear_session()
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=16, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save(cfg.MODEL_PATH)
    with open(cfg.ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    return model, encoder, accuracy, raw_rows, len(X_train)


def predict(model, encoder, hand):
    """Returns (gesture_name, confidence) for one hand."""
    return predict_features(model, encoder, normalize_landmarks(hand))


def predict_features(model, encoder, features):
    """Same as predict(), from already-normalized landmarks."""
    features = np.array([features], dtype=np.float32)
    probabilities = np.asarray(model(features, training=False))[0]   # much faster than model.predict() per frame
    best = int(np.argmax(probabilities))
    return encoder.classes_[best], float(probabilities[best])


# ---------- Gesture -> action mapping ----------

def load_mapping(known_gestures):
    """Saved mapping, cleaned up: only gestures the model knows, only valid actions,
    and each action used by at most one gesture. Missing gestures get a default."""
    try:
        with open(cfg.MAPPING_PATH) as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError):
        saved = {}

    mapping, used = {}, set()
    for source in (saved, cfg.DEFAULT_MAPPING):
        for gesture in known_gestures:
            action = source.get(gesture)
            if gesture in mapping or action not in ACTIONS:
                continue
            if action != "none" and action in used:
                continue
            mapping[gesture] = action
            used.add(action)
    for gesture in known_gestures:
        mapping.setdefault(gesture, "none")
    if NONE_LABEL in mapping:
        mapping[NONE_LABEL] = "none"   # "not a gesture" never does anything

    save_mapping(mapping)
    return mapping


def save_mapping(mapping):
    with open(cfg.MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
