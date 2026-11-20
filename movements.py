"""Dynamic gestures (movements): cutting them out of the live video, storing them,
and recognising them with DTW (Dynamic Time Warping) + nearest neighbour.

A movement is the path of the palm centre over time, plus how the hand size changes
(the hand looks bigger when it moves towards the camera). Paths are normalised so the
same movement matches whether it was big or small, fast or slow, anywhere in the frame.
"""
import json
import math
from collections import deque

import numpy as np

import config as cfg

NONE_MOVEMENT = "_not_a_movement"   # paths from corrected mistakes: "this is none of my movements"
ASPECT = 480 / 640                  # camera y-units -> x-units, so circles stay round


# ---------- One frame of a hand ----------

def hand_point(hand):
    """(x, y, size) of a hand: palm centre and wrist-to-middle-knuckle length, in camera x-units."""
    palm = [hand.landmark[i] for i in (0, 5, 9, 13, 17)]
    x = sum(p.x for p in palm) / 5
    y = sum(p.y for p in palm) / 5 * ASPECT
    wrist, knuckle = hand.landmark[0], hand.landmark[9]
    size = math.hypot(knuckle.x - wrist.x, (knuckle.y - wrist.y) * ASPECT)
    return x, y, max(size, 1e-4)


def is_open_hand(hand):
    """True if at least 4 of the 4 long fingers are stretched out (open palm, 'four', etc.).
    Checked from the hand's shape directly, so it works even when the pose model is unsure
    during a fast movement."""
    def dist(i, j):
        a, b = hand.landmark[i], hand.landmark[j]
        return math.hypot(a.x - b.x, (a.y - b.y) * ASPECT)
    extended = sum(dist(0, tip) > dist(0, joint) * 1.1 for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return extended >= cfg.MOVE_OPEN_FINGERS


def open_fraction(points):
    """Share of a movement's frames in which the hand was open."""
    return sum(1 for p in points if p[4]) / max(len(points), 1)


# ---------- Cutting movements out of the live video ----------

class MovementTracker:
    """Watches one hand. A movement starts when the hand moves clearly (speed in hand-sizes
    per second) and ends when it slows down again. Returns each finished movement once."""

    PREROLL = 3   # frames kept from just before the start, so the beginning isn't cut off

    def __init__(self, need_rest=True):
        self.need_rest = need_rest   # a movement must start from a still hand (off while recording)
        self.reset()

    def reset(self):
        self.recent_speeds = deque(maxlen=cfg.MOVE_REST_WINDOW)
        self.history = deque(maxlen=self.PREROLL)
        self.capture = None          # frames of the movement in progress
        self.window = deque(maxlen=cfg.MOVE_SPEED_FRAMES + 1)   # recent frames for measuring speed
        self.speed = 0.0
        self.from_rest = True        # did the movement in progress start from a still hand?
        self.slow_frames = 0
        self.missing = 0           # frames in a row without the hand
        self.last_problem = None
        self.last_movement = None

    @property
    def active(self):
        return self.capture is not None

    def update(self, hand, pose, t):
        """Feed one frame (pose = True if the hand is open this frame). Returns a finished
        movement as a list of (x, y, size, time, pose), or None."""
        x, y, size = hand_point(hand)
        self.missing = 0
        self.window.append((x, y, size, t))
        if len(self.window) == self.window.maxlen:   # only once the window is full (fewer frames = noisy)
            # Speed over the last few frames, not frame to frame: tracking jitter averages
            # out over the window while a real movement keeps adding up.
            px, py, psize, pt = self.window[0]
            dt = max(t - pt, 1e-3)
            across = math.hypot(x - px, y - py) / size                 # hand-sizes moved
            depth = cfg.MOVE_DEPTH_WEIGHT * abs(math.log(size / psize))  # towards/away from camera
            self.speed = math.hypot(across, depth) / dt
        point = (x, y, size, t, pose)

        if self.capture is None:
            self.history.append(point)
            if self.speed > cfg.MOVE_START_SPEED:
                self.capture = list(self.history)
                self.slow_frames = 0
                still = sum(s < cfg.MOVE_STILL_SPEED for s in self.recent_speeds)
                self.from_rest = still >= cfg.MOVE_REST_FRAMES   # hand was still just before
            elif len(self.window) == self.window.maxlen:   # a real speed (not the 0 before the window fills)
                self.recent_speeds.append(self.speed)
            return None

        self.capture.append(point)
        self.slow_frames = self.slow_frames + 1 if self.speed < cfg.MOVE_STOP_SPEED else 0
        too_long = t - self.capture[0][3] > cfg.MOVE_MAX_SECONDS
        if self.slow_frames < cfg.MOVE_STOP_FRAMES and not too_long:
            return None
        movement = self.capture[:len(self.capture) - max(self.slow_frames - 1, 0)]
        self.capture = None
        self.history.clear()
        self.history.append(point)
        self.recent_speeds.clear()   # the next movement needs its own still moment, not the old one
        return self._checked(movement)

    def lost(self):
        """Call when the hand isn't visible this frame. Fast movements blur, and MediaPipe often
        loses the hand for a few frames - so wait a little before giving up. Returns a finished
        movement if waiting is over and one was in progress."""
        self.missing += 1
        if self.missing <= cfg.MOVE_MAX_GAP:
            return None            # keep everything; if the hand reappears further on, that jump is speed
        if self.capture is not None:
            return self.finish()
        self.reset()
        return None

    def finish(self):
        """The hand disappeared (fast movements can blur or leave the frame): end the movement
        in progress with what was captured, instead of throwing it away."""
        movement, self.capture = self.capture, None
        self.reset()
        return self._checked(movement) if movement else None

    def _checked(self, movement):
        self.last_problem = movement_problem(movement)
        if self.last_problem is None and self.need_rest and not self.from_rest:
            self.last_problem = "didn't start from a still hand"
        self.last_movement = movement
        return None if self.last_problem else movement


def movement_problem(points):
    """Why a captured motion doesn't count as a movement, or None if it does."""
    if len(points) < 3 or points[-1][3] - points[0][3] < cfg.MOVE_MIN_SECONDS:
        return "too short"
    size0 = points[0][2]
    travel = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])) / size0
    depth = max(abs(math.log(p[2] / size0)) for p in points)
    if travel < cfg.MOVE_MIN_TRAVEL and depth < math.log(cfg.MOVE_MIN_DEPTH_RATIO):
        return f"too small (moved {travel:.1f} hand-sizes, need {cfg.MOVE_MIN_TRAVEL})"
    return None


def describe(points):
    """'0.6 s, 4.2 hand-sizes' - for the activity log."""
    size0 = points[0][2]
    travel = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])) / size0
    return f"{points[-1][3] - points[0][3]:.1f} s, {travel:.1f} hand-sizes"


def returns_to_start(points, end):
    """True if everything after `end` is the hand coming back towards where the movement
    started, in one steady motion (like the return after a swipe) - so it can be ignored.
    A zigzag or a wave goes somewhere new or back and forth instead, so it must be kept."""
    x0, y0 = points[0][0], points[0][1]
    gaps = [math.hypot(p[0] - x0, p[1] - y0) for p in points[end - 1:]]
    if len(gaps) < 3 or gaps[-1] > cfg.MOVE_RETURN_SHARE * gaps[0]:
        return False                                    # didn't come most of the way back
    steps = [b - a for a, b in zip(gaps, gaps[1:]) if abs(b - a) > 0.002]
    if steps and sum(st < 0 for st in steps) / len(steps) < 0.8:
        return False                                    # not mostly getting closer
    rest = points[end - 1:]
    travelled = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(rest, rest[1:]))
    return travelled <= cfg.MOVE_RETURN_STRAIGHT * gaps[0]   # straight back, not round a curve (circle)


# ---------- Comparing movements ----------

def normalize_path(points, n=None):
    """Path -> (n, 3) array: start at (0, 0), scaled to one size, depth as log hand-size
    change, resampled to n evenly spaced moments in time."""
    n = n or cfg.MOVE_POINTS
    P = np.array([p[:3] for p in points], dtype=np.float64)
    xy = (P[:, :2] - P[0, :2]) / P[0, 2]                        # in hand-sizes
    extent = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)     # big and small versions look alike
    depth = np.log(P[:, 2] / P[0, 2]) * cfg.MOVE_DEPTH_WEIGHT
    path = np.column_stack([xy / extent, depth])
    old, new = np.linspace(0, 1, len(path)), np.linspace(0, 1, n)
    return np.column_stack([np.interp(new, old, path[:, k]) for k in range(3)])


def dtw_distances(query, templates, band=None):
    """DTW distance from one normalised path (n, 3) to many (T, n, 3). Lower = more alike.
    DTW lines up matching parts even when one movement was faster or slower in places."""
    band = band or cfg.MOVE_DTW_BAND
    count, n = len(templates), len(query)
    cost = np.linalg.norm(query[None, :, None, :] - templates[:, None, :, :], axis=-1)   # (T, n, n)
    D = np.full((count, n + 1, n + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            best_previous = np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
            D[:, i, j] = cost[:, i - 1, j - 1] + best_previous
    return D[:, n, n] / (2 * n)


# ---------- Stored movements ----------

class MovementLibrary:
    """All recorded movements (saved in movement_data.json) and the recogniser that uses them."""

    def __init__(self, path=None):
        self.path = path or cfg.MOVEMENT_DATA_PATH
        self.load()

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.samples = [(s["label"], [tuple(p) for p in s["path"]]) for s in data.get("samples", [])]
        except (OSError, ValueError, KeyError, TypeError):
            self.samples = []
        self._rebuild()

    def save(self):
        data = {"samples": [{"label": label, "path": [[round(v, 5) for v in p[:3]] for p in path]}
                            for label, path in self.samples]}
        with open(self.path, "w") as f:
            json.dump(data, f)

    def _rebuild(self):
        self.labels_array = [label for label, _ in self.samples]
        self.lengths = [len(path) for _, path in self.samples]   # frames per recording
        self.templates = (np.stack([normalize_path(p) for _, p in self.samples])
                          if self.samples else np.zeros((0, cfg.MOVE_POINTS, 3)))
        self.threshold = self._reject_threshold()

    def _reject_threshold(self):
        """How far a new movement may be from its closest recording and still count.
        Set from your own data: a few times the usual gap between recordings of one movement."""
        gaps = []
        for i, label in enumerate(self.labels_array):
            same = [j for j, other in enumerate(self.labels_array) if other == label and j != i]
            if label != NONE_MOVEMENT and same:
                gaps.append(dtw_distances(self.templates[i], self.templates[same]).min())
        if len(gaps) < 2:
            return cfg.MOVE_REJECT_DEFAULT
        return max(cfg.MOVE_REJECT_FACTOR * float(np.median(gaps)), cfg.MOVE_REJECT_MIN)   # your natural variation

    def labels(self):
        return sorted({label for label in self.labels_array if label != NONE_MOVEMENT})

    def counts(self):
        counts = {}
        for label in self.labels_array:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def add(self, label, points):
        self.samples.append((label, [tuple(p[:3]) for p in points]))
        self._rebuild()
        self.save()

    def rename(self, old, new):
        self.samples = [(new if l == old else l, p) for l, p in self.samples]
        self._rebuild()
        self.save()

    def remove(self, label):
        self.samples = [(l, p) for l, p in self.samples if l != label]
        self._rebuild()
        self.save()

    def distances(self, points, try_shorter=False):
        """Closest DTW distance to each movement name, closest first.
        try_shorter: also compare just the beginning of the movement, ending at several points,
        so a swipe still matches when the hand comes straight back afterwards."""
        if not self.samples:
            return {}
        ends = [len(points)]
        if try_shorter and len(points) > 8:
            shortest = max(4, int(min(self.lengths) * cfg.MOVE_SHORTEST_PART))
            step = max(1, (len(points) - shortest) // cfg.MOVE_END_TRIES)
            ends += [end for end in range(shortest, len(points), step) if returns_to_start(points, end)]
        best = {}
        for end in ends:
            d = dtw_distances(normalize_path(points[:end]), self.templates)
            for label, value in zip(self.labels_array, d):
                best[label] = min(best.get(label, np.inf), float(value))
        return dict(sorted(best.items(), key=lambda item: item[1]))

    def recognize(self, points):
        """(name, distance) of the closest recorded movement, or (None, distance) if it's
        too far from everything or closest to a corrected mistake."""
        distances = self.distances(points, try_shorter=True)
        if not distances:
            return None, None
        ranked = list(distances.items())
        name, distance = ranked[0]
        if name == NONE_MOVEMENT or distance > self.threshold:
            return None, distance
        runner_up = next((d for n, d in ranked[1:] if n != NONE_MOVEMENT), None)
        if runner_up is not None and distance > cfg.MOVE_AMBIGUITY_RATIO * runner_up:
            return None, distance    # almost as close to another movement: too unsure to act
        return name, distance

    def similarity_warning(self, name, points):
        """(level, message) if a new recording is closer to another movement than to its own, else None."""
        distances = {k: v for k, v in self.distances(points).items() if k != NONE_MOVEMENT}
        own = distances.pop(name, None)
        if not distances:
            return None
        other, distance = next(iter(distances.items()))
        if own is not None and distance < own:
            return "strong", (f"This is closer to '{other}' ({distance:.2f}) than to your other "
                              f"'{name}' recordings ({own:.2f}). Retake it more clearly.")
        if distance < self.threshold * 0.6:
            return "strong", f"This looks almost the same as '{other}' ({distance:.2f}). Make it more different."
        if distance < self.threshold:
            return "mild", f"This is quite similar to '{other}' ({distance:.2f}). It may sometimes be confused."
        return None
