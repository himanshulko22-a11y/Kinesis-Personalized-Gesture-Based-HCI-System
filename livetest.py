"""Live test: scores the pose model on fresh hand poses made right now.

Kinesis asks for each pose in turn ("Make: fist  3, 2, 1"), then watches for a few seconds.
Every frame's prediction is compared with the pose that was asked for. Nothing is saved
as training data and no actions run, so this is a clean, real-world accuracy measurement
(today's light, position and hand) - unlike Evaluate, which tests old recordings.
"""
import collections
import csv
import os
import random
import time

import config as cfg

UNSURE = "(unsure)"   # the model's best guess was below the confidence threshold


class LiveTest:
    def __init__(self, poses, threshold, shuffle=True, seed=None):
        self.poses = list(poses)
        if shuffle:
            random.Random(seed).shuffle(self.poses)   # so you can't just stay in one position
        self.threshold = threshold
        self.step = 0
        self.phase = "ready"            # "ready" (countdown) -> "hold" (scoring) -> next pose
        self.phase_start = None
        self.frames = collections.Counter()       # pose -> frames scored with a hand in view
        self.correct = collections.Counter()      # pose -> frames recognised correctly (as in real use)
        self.no_hand = collections.Counter()      # pose -> frames with no hand in view
        self.guesses = collections.defaultdict(collections.Counter)   # pose -> what was predicted
        self.current_guess = None

    @property
    def finished(self):
        return self.step >= len(self.poses)

    @property
    def target(self):
        return None if self.finished else self.poses[self.step]

    def seconds_left(self, now):
        length = cfg.LIVE_TEST_READY if self.phase == "ready" else cfg.LIVE_TEST_HOLD
        return max(0.0, length - (now - self.phase_start))

    def update(self, now, prediction):
        """Feed one frame. prediction = (pose, confidence) for the hand in view, or None if no hand."""
        if self.finished:
            return
        if self.phase_start is None:
            self.phase_start = now
        if self.phase == "ready":
            if now - self.phase_start >= cfg.LIVE_TEST_READY:
                self.phase, self.phase_start = "hold", now
            return
        target = self.target
        if prediction is None:
            self.no_hand[target] += 1
            self.current_guess = None
        else:
            pose, confidence = prediction
            guess = pose if confidence >= self.threshold else UNSURE
            self.frames[target] += 1
            self.guesses[target][guess] += 1
            self.correct[target] += guess == target
            self.current_guess = guess
        if now - self.phase_start >= cfg.LIVE_TEST_HOLD:
            self.step += 1
            self.phase, self.phase_start, self.current_guess = "ready", now, None

    def results(self):
        """[(accuracy or None, pose, frames, most common wrong guess or None, its share)], weakest first."""
        rows = []
        for pose in self.poses:
            n = self.frames[pose]
            wrong = [(guess, count) for guess, count in self.guesses[pose].most_common() if guess != pose]
            top = wrong[0] if wrong else (None, 0)
            rows.append((self.correct[pose] / n if n else None, pose, n, top[0], top[1] / n if n else 0.0))
        return sorted(rows, key=lambda row: -1 if row[0] is None else row[0])

    def overall(self):
        total = sum(self.frames.values())
        return sum(self.correct.values()) / total if total else None

    def matrix(self):
        """Confusion matrix over the tested poses plus UNSURE (rows: asked for, columns: predicted)."""
        names = sorted(self.poses) + [UNSURE]
        index = {name: i for i, name in enumerate(names)}
        grid = [[0] * len(names) for _ in names]
        for target, guesses in self.guesses.items():
            for guess, count in guesses.items():
                if guess in index:   # a pose outside this test (e.g. 'not a gesture') counts as unsure
                    grid[index[target]][index[guess]] += count
                else:
                    grid[index[target]][index[UNSURE]] += count
        return names, grid

    def save(self, path, note=""):
        """Appends one row per pose to a CSV, so several runs (or people) can be compared in the report."""
        new_file = not os.path.exists(path)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "note", "pose", "frames", "accuracy", "most_confused_with", "confused_share",
                                 "no_hand_frames", "threshold"])
            for accuracy, pose, n, wrong, share in self.results():
                writer.writerow([stamp, note, pose, n, "" if accuracy is None else round(accuracy, 4),
                                 wrong or "", round(share, 4), self.no_hand[pose], self.threshold])
