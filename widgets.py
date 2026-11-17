"""Small reusable pieces of the Kinesis window."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from actions import ACTIONS


# ---------- Small UI helpers ----------

def section_title(text):
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


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

class GestureMappingRow(QWidget):
    def __init__(self, gesture, action, on_change, on_record_more, on_delete):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        name = QLabel(gesture)
        name.setMinimumWidth(90)
        layout.addWidget(name)

        self.combo = QComboBox()
        for key, info in ACTIONS.items():
            self.combo.addItem(info.label, userData=key)
        self.combo.setCurrentIndex(max(0, self.combo.findData(action)))
        self.combo.currentIndexChanged.connect(lambda: on_change(gesture, self.combo.currentData()))
        layout.addWidget(self.combo)

        more = button("+", lambda: on_record_more(gesture), f"Record more samples for '{gesture}'")
        more.setFixedWidth(34)
        layout.addWidget(more)

        delete = button("Delete", lambda: on_delete(gesture), name="deleteBtn")
        delete.setFixedWidth(85)
        layout.addWidget(delete)


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
