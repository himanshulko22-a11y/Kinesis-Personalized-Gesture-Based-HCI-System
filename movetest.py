"""Movement Live Test: scores the DTW movement recogniser on fresh movements made right now.

Kinesis asks for each recorded movement several times in a random order ("Move: swipe_left  3, 2, 1, GO").
Each attempt is cut out of the video and recognised exactly as in normal use (same tracker, same
open-hand rule, same DTW nearest neighbour and reject rules), then logged as:
  correct       - recognised as the movement that was asked for
  wrong         - recognised as a different movement
  rejected      - Kinesis refused to name it (too far from every recording, too close to two, a corrected
                  mistake, or the hand wasn't open) - in real use nothing would happen
  not detected  - no movement was seen before the time ran out

The test only READS the movement library. Attempts are never added to movement_data.json, so the
recordings it compares against stay the same and the score isn't inflated by testing on itself.
"""
import collections
import csv
import os
import random
import time

import config as cfg
import movements as mv

CORRECT, WRONG, REJECTED, MISSED = "correct", "wrong", "rejected", "not detected"
REJECTED_COL = "(rejected)"       # confusion-table columns for attempts that weren't named
MISSED_COL = "(not detected)"


def rejection_reason(library, movement):
    """Why recognize() returned None for this movement (same wording as the activity log)."""
    ranked = list(library.distances(movement, try_shorter=True).items())
    if not ranked:
        return "no movements recorded"
    if ranked[0][0] == mv.NONE_MOVEMENT:
        return "matched a corrected mistake"
    if ranked[0][1] > library.threshold:
        return f"closest '{ranked[0][0]}' {ranked[0][1]:.3f} > limit {library.threshold:.3f}"
    second = next(((n, d) for n, d in ranked[1:] if n != mv.NONE_MOVEMENT), ("?", 0))
    return f"unsure between '{ranked[0][0]}' {ranked[0][1]:.3f} and '{second[0]}' {second[1]:.3f}"


class MovementLiveTest:
    def __init__(self, library, names, reps, shuffle=True, seed=None):
        self.library = library          # read only - nothing is ever added to it here
        self.names = sorted(names)
        self.reps = reps
        rng = random.Random(seed)
        self.schedule = []              # every movement once per round, rounds in a random order
        for _ in range(reps):
            round_ = list(self.names)
            if shuffle:
                rng.shuffle(round_)
                if self.schedule and len(round_) > 1 and round_[0] == self.schedule[-1]:
                    round_[0], round_[1] = round_[1], round_[0]   # never the same one twice in a row
            self.schedule += round_
        self.trackers = {"Left": mv.MovementTracker(), "Right": mv.MovementTracker()}
        self.attempts = []              # one dict per attempt, in order
        self.step = 0
        self.phase = "ready"            # "ready" (countdown) -> "go" (waiting for the movement) -> "result"
        self.phase_start = None
        self.hint = None                # e.g. "too small - try again, a bit bigger"
        self.started = time.time()

    @property
    def finished(self):
        return self.step >= len(self.schedule)

    @property
    def target(self):
        return None if self.finished else self.schedule[self.step]

    @property
    def last(self):
        return self.attempts[-1] if self.attempts else None

    def capture(self):
        """The movement being made right now (for drawing), or None."""
        for tracker in self.trackers.values():
            if tracker.active and tracker.capture and len(tracker.capture) > 1:
                return tracker.capture
        return None

    def seconds_left(self, now):
        length = {"ready": cfg.MOVE_TEST_READY, "go": cfg.MOVE_TEST_TIMEOUT, "result": cfg.MOVE_TEST_SHOW}[self.phase]
        return max(0.0, length - (now - self.phase_start))

    def _feed(self, hands, now):
        """Feeds every hand to its tracker (as in normal use). Returns the first finished movement, or None."""
        seen = {label for label, _ in hands}
        finished = None
        for label, tracker in self.trackers.items():
            if label not in seen:
                movement = tracker.lost()
                finished = finished or movement
        for label, landmarks in hands:
            tracker = self.trackers.get(label)
            if tracker is None:
                continue
            movement = tracker.update(landmarks, mv.is_open_hand(landmarks), now)
            finished = finished or movement
            if movement is None and tracker.last_movement is not None:
                if self.phase == "go" and mv.open_fraction(tracker.last_movement) >= cfg.MOVE_OPEN_SHARE:
                    self.hint = f"not counted ({tracker.last_problem}) - hold still, then try again"
                tracker.last_movement = None
        return finished

    def update(self, now, hands):
        """Feed one frame. hands = [(label, landmarks)] from the hand tracker."""
        if self.finished:
            return
        if self.phase_start is None:
            self.phase_start = now
        elapsed = now - self.phase_start

        if self.phase in ("ready", "result"):
            self._feed(hands, now)   # keeps the trackers warm (they need a still hand before a movement);
            #                          anything finished now (e.g. the hand coming back) is thrown away
            if self.phase == "result" and elapsed >= cfg.MOVE_TEST_SHOW:
                self.step += 1
                self.phase, self.phase_start, self.hint = "ready", now, None
            elif self.phase == "ready" and elapsed >= cfg.MOVE_TEST_READY:
                for tracker in self.trackers.values():
                    if tracker.active:   # started moving before GO: that one doesn't count
                        tracker.reset()
                self.phase, self.phase_start, self.hint = "go", now, None
            return

        movement = self._feed(hands, now)
        if movement is not None:
            self._score(movement, now)
            return
        moving = any(tracker.active for tracker in self.trackers.values())
        if elapsed >= cfg.MOVE_TEST_TIMEOUT and (not moving or elapsed >= cfg.MOVE_TEST_TIMEOUT + cfg.MOVE_MAX_SECONDS):
            self._record(MISSED, None, None, self.hint or "no movement seen in time", None, now)

    def _score(self, movement, now):
        if cfg.MOVEMENT_NEEDS_OPEN_HAND and mv.open_fraction(movement) < cfg.MOVE_OPEN_SHARE:
            self._record(REJECTED, None, None, "hand wasn't open", movement, now)
            return
        name, distance = self.library.recognize(movement)
        if name is None:
            self._record(REJECTED, None, distance, rejection_reason(self.library, movement), movement, now)
        elif name == self.target:
            self._record(CORRECT, name, distance, "", movement, now)
        else:
            self._record(WRONG, name, distance, "", movement, now)

    def _record(self, result, seen_as, distance, reason, movement, now):
        self.attempts.append({"number": self.step + 1, "asked": self.target, "result": result, "seen_as": seen_as,
                              "distance": distance, "reason": reason, "path": movement,
                              "seconds": None if movement is None else movement[-1][3] - movement[0][3]})
        self.phase, self.phase_start = "result", now

    # ---------- Results ----------

    def counts(self, name=None):
        """Counter of results for one movement, or for all of them."""
        return collections.Counter(a["result"] for a in self.attempts if name is None or a["asked"] == name)

    def results(self):
        """[(accuracy or None, name, attempts, counts, most common wrong movement or None, how often)], weakest first."""
        rows = []
        for name in self.names:
            c = self.counts(name)
            n = sum(c.values())
            wrong = collections.Counter(a["seen_as"] for a in self.attempts if a["asked"] == name and a["result"] == WRONG)
            top = wrong.most_common(1)[0] if wrong else (None, 0)
            rows.append((c[CORRECT] / n if n else None, name, n, c, top[0], top[1]))
        return sorted(rows, key=lambda row: -1 if row[0] is None else row[0])

    def overall(self):
        n = len(self.attempts)
        return self.counts()[CORRECT] / n if n else None

    def rate(self, result):
        n = len(self.attempts)
        return self.counts()[result] / n if n else None

    def matrix(self):
        """Rows: movement asked for. Columns: what it was recognised as, plus rejected / not detected."""
        names = list(self.names) + [REJECTED_COL, MISSED_COL]
        index = {name: i for i, name in enumerate(names)}
        grid = [[0] * len(names) for _ in names]
        for a in self.attempts:
            column = {REJECTED: REJECTED_COL, MISSED: MISSED_COL}.get(a["result"], a["seen_as"])
            if column not in index:      # a movement outside this test - count it as rejected
                column = REJECTED_COL
            grid[index[a["asked"]]][index[column]] += 1
        return names, grid

    def save(self, path, tracking, stopped=False):
        """Appends this run to the results CSV, in the same layout as the pose Live Test file: one OVERALL row,
        then one row per movement (weakest first). Only the newest runs are kept (the latest and the one
        before). Every row carries the run number, so runs are easy to compare in Excel. Returns the run number."""
        run = max(_runs(path), default=0) + 1
        now = time.localtime()
        base = [run, time.strftime("%Y-%m-%d", now), time.strftime("%H:%M", now), tracking]
        end = [round(self.library.threshold, 3), "yes" if stopped else "no"]
        c, n = self.counts(), len(self.attempts)
        rows = [base + ["OVERALL", _pct(self.overall()), n, c[CORRECT], c[WRONG], c[REJECTED], c[MISSED],
                        _pct(self.rate(REJECTED)), "", ""] + end]
        for accuracy, name, count, cs, wrong, times in self.results():
            rows.append(base + [name, _pct(accuracy), count, cs[CORRECT], cs[WRONG], cs[REJECTED], cs[MISSED],
                                _pct(cs[REJECTED] / count) if count else "", wrong or "",
                                _pct(times / count) if wrong else ""] + end)
        _append(path, COLUMNS, rows)
        _keep_newest_runs(path, COLUMNS, cfg.MOVE_TEST_KEEP_RUNS)
        return run


COLUMNS = ["run", "date", "time", "tracking", "movement", "accuracy_%", "attempts", "correct", "wrong", "rejected",
           "not_detected", "rejected_%", "most_confused_with", "confused_%", "reject_limit", "stopped_early"]


def _pct(value):
    return "" if value is None else round(value * 100, 1)


def _runs(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            return set()   # older layout: it is replaced on save
        return {int(row["run"]) for row in reader if row.get("run", "").isdigit()}


def _append(path, columns, rows):
    if os.path.exists(path):
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != columns:   # a file from an older layout: start it again
            os.remove(path)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(columns)
        writer.writerows(rows)


def _keep_newest_runs(path, columns, keep):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    newest = sorted({int(row["run"]) for row in rows})[-keep:]
    kept = [row for row in rows if int(row["run"]) in newest]
    if len(kept) == len(rows):
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept)
