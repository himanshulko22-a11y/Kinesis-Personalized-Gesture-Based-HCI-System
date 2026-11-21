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

    def save(self, path, tracking, stopped=False):
        """Appends this run to a CSV: one OVERALL row, then one row per pose (weakest first). Every row carries
        the run number, so runs are easy to tell apart and to compare (filter or pivot on "run" in Excel)."""
        upgrade_old_format(path)
        runs = read_runs(path)
        run = max(runs, default=0) + 1
        now = time.localtime()
        base = [run, time.strftime("%Y-%m-%d", now), time.strftime("%H:%M", now), tracking]
        total = sum(self.frames.values())
        rows = [base + ["OVERALL", _pct(self.overall()), total, sum(self.correct.values()),
                        sum(self.no_hand.values()), "", "", _pct(self.threshold), "yes" if stopped else "no"]]
        for accuracy, pose, n, wrong, share in self.results():
            rows.append(base + [pose, _pct(accuracy), n, self.correct[pose], self.no_hand[pose], wrong or "",
                                _pct(share) if wrong else "", _pct(self.threshold), "yes" if stopped else "no"])
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(COLUMNS)
            writer.writerows(rows)
        keep_newest_runs(path, cfg.LIVE_TEST_KEEP_RUNS)
        return run


COLUMNS = ["run", "date", "time", "tracking", "pose", "accuracy_%", "frames", "correct_frames", "no_hand_frames",
           "most_confused_with", "confused_%", "threshold_%", "stopped_early"]


def _pct(value):
    return "" if value is None else round(value * 100, 1)


def read_runs(path):
    """Run numbers already in the file."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {int(row["run"]) for row in csv.DictReader(f) if row.get("run", "").isdigit()}


def keep_newest_runs(path, keep):
    """Deletes all but the newest `keep` runs from the results file."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    newest = sorted({int(row["run"]) for row in rows})[-keep:]
    kept = [row for row in rows if int(row["run"]) in newest]
    if len(kept) == len(rows):
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(kept)


def upgrade_old_format(path):
    """Rewrites a results file from the first version (no run numbers) into the current layout."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "run" in reader.fieldnames:
            return
        old = list(reader)
    rows, runs = [], {}
    for row in old:
        runs.setdefault(row["time"], []).append(row)
    for run, (stamp, items) in enumerate(runs.items(), start=1):
        date, clock = stamp.split(" ")[0], stamp.split(" ")[-1][:5]
        tracking = items[0]["note"].split(",")[0].replace(" tracking", "")
        stopped = "yes" if "stopped" in items[0]["note"] else "no"
        threshold = _pct(float(items[0]["threshold"]))
        frames = sum(int(r["frames"]) for r in items)
        correct = sum(round(float(r["accuracy"] or 0) * int(r["frames"])) for r in items)
        base = [run, date, clock, tracking]
        rows.append(base + ["OVERALL", _pct(correct / frames) if frames else "", frames, correct,
                            sum(int(r["no_hand_frames"]) for r in items), "", "", threshold, stopped])
        for r in sorted(items, key=lambda r: float(r["accuracy"]) if r["accuracy"] else -1):
            n = int(r["frames"])
            acc = float(r["accuracy"]) if r["accuracy"] else None
            rows.append(base + [r["pose"], _pct(acc), n, round((acc or 0) * n), r["no_hand_frames"],
                                r["most_confused_with"], _pct(float(r["confused_share"])) if r["most_confused_with"] else "",
                                threshold, stopped])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
