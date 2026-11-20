"""Small reusable pieces of the Kinesis window."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config as cfg
from actions import ACTIONS


# ---------- Small UI helpers ----------

def field_label(text):
    """Small label above a control inside a section (e.g. a slider's name)."""
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


class CollapsibleSection(QFrame):
    """A card with a title you can click to show or hide what's inside."""

    def __init__(self, title, expanded=True, on_toggle=None, danger=False):
        super().__init__()
        self.setObjectName("dangerCard" if danger else "sectionCard")
        self.title, self.on_toggle = title, on_toggle
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = QPushButton()
        self.header.setObjectName("dangerHeader" if danger else "sectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.clicked.connect(self.toggle)
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(14, 0, 14, 12)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.expanded = expanded
        self.body.setVisible(expanded)
        # '&' would be read as a keyboard-shortcut marker by QPushButton, so it is doubled
        self.header.setText(("▾   " if expanded else "▸   ") + self.title.replace("&", "&&"))

    def toggle(self):
        self.set_expanded(not self.expanded)
        if self.on_toggle:
            self.on_toggle(self.title, self.expanded)


class HandTile(QFrame):
    """Live view of one hand: its gesture, how sure the model is, and what it does."""

    def __init__(self, side):
        super().__init__()
        self.setObjectName("handTile")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(4)
        caption = QLabel(f"{side.upper()} HAND")
        caption.setObjectName("tileCaption")
        self.name = QLabel("—")
        self.name.setObjectName("tileGesture")
        self.bar = QProgressBar()
        self.bar.setObjectName("confBar")
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.detail = QLabel(" ")
        self.detail.setObjectName("tileDetail")
        for widget in (caption, self.name, self.bar, self.detail):
            layout.addWidget(widget)
        self._shown = None
        self._level = None

    def show_hand(self, name, confidence, detail, level):
        """level: "good" (confident), "low" (below the threshold), "move" (a movement), or None (no hand)."""
        shown = (name, round(confidence * 100), detail, level)
        if shown == self._shown:
            return   # nothing changed: skip re-styling (runs every frame)
        self._shown = shown
        self.name.setText(name or "—")
        self.bar.setValue(round(confidence * 100))
        self.detail.setText(detail or " ")
        if level != self._level:
            self._level = level
            for widget in (self, self.name, self.bar):
                widget.setProperty("level", level or "none")
                widget.style().unpolish(widget)
                widget.style().polish(widget)


def note(text):
    label = QLabel(text)
    label.setObjectName("note")
    label.setWordWrap(True)
    return label


def button(text, on_click, tooltip=None, name=None):
    btn = QPushButton(text)
    btn.clicked.connect(lambda: on_click())   # lambda drops Qt's 'checked' argument
    if tooltip:
        btn.setToolTip(tooltip)
    if name:
        btn.setObjectName(name)
    return btn


def clear_layout(layout):
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget:
            widget.hide()
            widget.deleteLater()


def ask(parent, title, text):
    return QMessageBox.question(parent, title, text, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes


# ---------- Dialog / row widgets ----------

class SelectGesturesDialog(QDialog):
    """Tick several gestures and delete them together (poses are retrained only once)."""

    def __init__(self, parent, poses, movements, restore=False):
        super().__init__(parent)
        self.restore = restore
        self.setWindowTitle("Restore Deleted Gestures" if restore else "Delete Several Gestures")
        self.resize(380, 520)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(note(
            "Tick what you want back. A restored pose works straight away if the model still knows it; "
            f"otherwise click Retrain Model. Deleted gestures are kept for {cfg.TRASH_KEEP_DAYS} days." if restore else
            "Tick everything you want to delete. It's instant - no retraining now. Deleted poses are "
            "ignored until you next click Retrain Model, and everything can be restored later."))
        self.list = QListWidget()
        for title, names, kind in (("STATIC POSES", poses, "pose"), ("MOVEMENTS", movements, "movement")):
            if not names:
                continue
            header = QListWidgetItem(title)
            header.setFlags(Qt.NoItemFlags)   # a heading, not something to tick
            self.list.addItem(header)
            for name in names:
                item = QListWidgetItem(name)
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, kind)
                self.list.addItem(item)
        self.list.itemChanged.connect(self._update_count)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        for text, state in (("Select all", Qt.Checked), ("Select none", Qt.Unchecked)):
            link = button(text, lambda state=state: self._set_all(state), name="linkBtn")
            row.addWidget(link)
        row.addStretch()
        layout.addLayout(row)
        row = QHBoxLayout()
        self.delete_btn = button("Restore" if restore else "Delete", self.accept,
                                 name=None if restore else "deleteBtn")
        row.addWidget(self.delete_btn)
        row.addWidget(button("Cancel", self.reject, name="secondaryBtn"))
        layout.addLayout(row)
        self._update_count()

    def _items(self):
        items = (self.list.item(i) for i in range(self.list.count()))
        return [item for item in items if item.flags() & Qt.ItemIsUserCheckable]

    def _set_all(self, state):
        for item in self._items():
            item.setCheckState(state)

    def _update_count(self, *_):
        count = len(self.selected())
        verb = "Restore" if self.restore else "Delete"
        self.delete_btn.setText(f"{verb} {count} selected" if count else verb)
        self.delete_btn.setEnabled(count > 0)

    def selected(self):
        """[(name, "pose" or "movement"), ...] that are ticked."""
        return [(item.text(), item.data(Qt.UserRole)) for item in self._items() if item.checkState() == Qt.Checked]

class GestureMappingRow(QWidget):
    def __init__(self, gesture, action, on_change, on_record_more, on_delete, on_rename=None, movement=False):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        name = QLabel(gesture)
        name.setObjectName("mappingName")
        name.setFixedWidth(110)
        layout.addWidget(name)

        self.combo = QComboBox()
        for key, info in ACTIONS.items():
            if movement and info.kind in ("cursor", "hold", "hold_move"):
                continue   # a movement is over in a moment, so it can't steer the cursor or hold a button
            self.combo.addItem(info.label, userData=key)
        self.combo.setCurrentIndex(max(0, self.combo.findData(action)))
        self.combo.currentIndexChanged.connect(lambda: on_change(gesture, self.combo.currentData()))
        layout.addWidget(self.combo, 1)

        icons = [("+", lambda: on_record_more(gesture), "iconBtn",
                  f"Record another '{gesture}' movement" if movement else f"Record more samples for '{gesture}'")]
        if on_rename is not None:
            icons.append(("✎", lambda: on_rename(gesture), "iconBtn", f"Rename '{gesture}'"))
        icons.append(("✕", lambda: on_delete(gesture), "iconDeleteBtn", f"Delete '{gesture}'"))
        for text, on_click, style, tip in icons:
            icon = button(text, on_click, tip, name=style)
            icon.setFixedSize(30, 30)
            icon.setCursor(Qt.PointingHandCursor)
            layout.addWidget(icon)


class ConfusionMatrixDialog(QDialog):
    def __init__(self, class_names, matrix, parent=None):
        super().__init__(parent)
        n = len(class_names)
        self.setWindowTitle("Confusion Matrix")
        self.resize(min(120 + 80 * n, 800), min(160 + 40 * n, 600))
        layout = QVBoxLayout(self)
        layout.addWidget(note("Rows = actual gesture, columns = predicted gesture. Diagonal = correct."))

        table = QTableWidget(n, n)
        table.setHorizontalHeaderLabels(list(class_names))
        table.setVerticalHeaderLabels(list(class_names))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        row_max = matrix.max(axis=1, keepdims=True).clip(min=1)
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(str(int(matrix[i, j])))
                item.setTextAlignment(Qt.AlignCenter)
                strength = matrix[i, j] / row_max[i, 0]
                if i == j:   # green = correct, red = confused
                    color = QColor(int(60 - strength * 20), int(60 + strength * 120), int(70 - strength * 20))
                else:
                    color = QColor(int(60 + strength * 150), int(60 - strength * 20), int(70 - strength * 20))
                item.setBackground(color)
                item.setForeground(QColor("#ffffff"))
                table.setItem(i, j, item)
        layout.addWidget(table)
        layout.addWidget(button("Close", self.accept))
