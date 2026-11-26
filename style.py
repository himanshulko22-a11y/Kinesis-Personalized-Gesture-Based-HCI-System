"""Dark purple theme for the Kinesis window.

Colours: background #120c20, cards #1b1430, raised #241a3d, borders #2f2550,
accent purple #7c4dff, text #ece6fa, muted #9d8ec2, good #5ee39a, warning #ffb86b, danger #ff7a93.
"""

STYLESHEET = """
/* ---------- Base ---------- */
QWidget {
    color: #ece6fa;
    font-family: "Segoe UI", Arial;
    font-size: 13px;
}
QMainWindow, QWidget#root, QWidget#panel, QScrollArea#panelScroll, QDialog, QMessageBox {
    background-color: #120c20;
}
QScrollArea#panelScroll { border: none; }
QLabel, QCheckBox { background: transparent; }
QToolTip {
    background-color: #241a3d;
    color: #ece6fa;
    border: 1px solid #7c4dff;
    border-radius: 6px;
    padding: 6px;
}

/* ---------- Header ---------- */
QLabel#appTitle {
    font-size: 26px;
    font-weight: 800;
    color: #b794ff;
}
QLabel#tagline {
    font-size: 12px;
    color: #8a7bb0;
}
QLabel#statusPill {
    padding: 5px 14px;
    border-radius: 13px;
    font-size: 12px;
    font-weight: bold;
    color: #5ee39a;
    background-color: rgba(94, 227, 154, 0.12);
    border: 1px solid rgba(94, 227, 154, 0.45);
}
QLabel#statusPill[state="paused"] {
    color: #ffb86b; background-color: rgba(255, 184, 107, 0.12); border-color: rgba(255, 184, 107, 0.5);
}
QLabel#statusPill[state="locked"] {
    color: #ff7a93; background-color: rgba(255, 122, 147, 0.12); border-color: rgba(255, 122, 147, 0.5);
}
QLabel#statusPill[state="rec"] {
    color: #ffffff; background-color: #e5484d; border-color: #e5484d;
}
QLabel#statusPill[state="review"] {
    color: #b794ff; background-color: rgba(124, 77, 255, 0.15); border-color: rgba(124, 77, 255, 0.6);
}

QPushButton#pinBtn {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0px;
    font-size: 16px;
    color: #c9b3ff;
}
QPushButton#pinBtn:hover { background-color: #241a3d; border-color: #3a2e63; }

/* ---------- Video ---------- */
QFrame#videoFrame {
    background-color: #0b0716;
    border: 1px solid #2f2550;
    border-radius: 16px;
}
QLabel#videoLabel {
    color: #b794ff;
}

/* ---------- Live card and hand tiles ---------- */
QFrame#liveCard {
    background-color: #1b1430;
    border: 1px solid #2f2550;
    border-radius: 14px;
}
QFrame#handTile {
    background-color: #241a3d;
    border: 1px solid #2f2550;
    border-radius: 10px;
}
QFrame#handTile[level="good"] { border-color: rgba(94, 227, 154, 0.55); }
QFrame#handTile[level="move"] { border-color: #7c4dff; background-color: #2a1d4d; }
QLabel#tileCaption {
    font-size: 10px;
    font-weight: bold;
    color: #8a7bb0;
}
QLabel#tileGesture {
    font-size: 20px;
    font-weight: bold;
    color: #6f6290;
}
QLabel#tileGesture[level="good"] { color: #ffffff; }
QLabel#tileGesture[level="low"] { color: #b3a6d6; }
QLabel#tileGesture[level="move"] { color: #d2bcff; }
QLabel#tileDetail {
    font-size: 11px;
    color: #9d8ec2;
}
QProgressBar#confBar {
    background-color: #160f29;
    border: none;
    border-radius: 3px;
}
QProgressBar#confBar::chunk { border-radius: 3px; background-color: #4a3f6b; }
QProgressBar#confBar[level="good"]::chunk { background-color: #5ee39a; }
QProgressBar#confBar[level="low"]::chunk { background-color: #ffb86b; }
QProgressBar#confBar[level="move"]::chunk { background-color: #7c4dff; }
QLabel#gestureLabel {
    font-size: 13px;
    font-weight: bold;
    color: #ece6fa;
}
QLabel#confidenceLabel {
    font-size: 12px;
    color: #8a7bb0;
}
QLabel#lastActionLabel {
    font-size: 12px;
    color: #5ee39a;
}
QLabel#statusLabel {
    color: #d9cff5;
}

/* ---------- Section cards ---------- */
QFrame#sectionCard, QFrame#dangerCard {
    background-color: #1b1430;
    border: 1px solid #2f2550;
    border-radius: 12px;
}
QFrame#dangerCard { border-color: #4a2233; }
QPushButton#sectionHeader, QPushButton#dangerHeader {
    background-color: transparent;
    color: #c9b3ff;
    font-size: 14px;
    font-weight: bold;
    text-align: left;
    padding: 12px 14px;
    border: none;
    border-radius: 12px;
}
QPushButton#dangerHeader { color: #ff7a93; }
QPushButton#sectionHeader:hover, QPushButton#dangerHeader:hover {
    background-color: #241a3d;
}
QLabel#fieldLabel {
    font-size: 12px;
    font-weight: bold;
    color: #b3a6d6;
}
QLabel#groupLabel {
    font-size: 11px;
    font-weight: bold;
    color: #8a7bb0;
    padding: 8px 0px 2px 0px;
}
QLabel#mappingName { color: #ece6fa; }
QLabel#pendingLabel {
    color: #ffb86b;
    font-size: 12px;
    background-color: rgba(255, 184, 107, 0.08);
    border: 1px solid rgba(255, 184, 107, 0.35);
    border-radius: 8px;
    padding: 6px 8px;
}
QLabel#note {
    color: #8a7bb0;
    font-size: 11px;
}

/* ---------- Buttons ---------- */
QPushButton {
    background-color: #7c4dff;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover { background-color: #9068ff; }
QPushButton:pressed { background-color: #6a3df0; }
QPushButton:disabled { background-color: #2f2550; color: #6f6290; }
QPushButton#secondaryBtn {
    background-color: #241a3d;
    color: #d2bcff;
    border: 1px solid #3a2e63;
}
QPushButton#secondaryBtn:hover { background-color: #2f2252; border-color: #7c4dff; }
QPushButton#deleteBtn {
    background-color: #3a1626;
    color: #ff9eb0;
    border: 1px solid #5a2236;
}
QPushButton#deleteBtn:hover { background-color: #4e1d33; }
QPushButton#linkBtn {
    background: transparent;
    color: #9d8ec2;
    padding: 2px 6px;
    font-weight: normal;
    font-size: 12px;
}
QPushButton#linkBtn:hover { color: #d2bcff; }
QPushButton#iconBtn, QPushButton#iconDeleteBtn {
    padding: 0px;
    font-size: 15px;
    border-radius: 7px;
    background-color: #241a3d;
    color: #c9b3ff;
    border: 1px solid #3a2e63;
}
QPushButton#iconBtn:hover { background-color: #7c4dff; color: white; border-color: #7c4dff; }
QPushButton#iconDeleteBtn { color: #ff7a93; }
QPushButton#iconDeleteBtn:hover { background-color: #4e1d33; border-color: #ff7a93; }

/* ---------- Inputs ---------- */
QComboBox, QLineEdit, QKeySequenceEdit QLineEdit {
    background-color: #160f29;
    border: 1px solid #3a2e63;
    border-radius: 7px;
    padding: 5px 8px;
    color: #ece6fa;
    min-height: 18px;
}
QComboBox { min-width: 130px; }
QComboBox:hover, QLineEdit:focus, QKeySequenceEdit QLineEdit:focus { border-color: #7c4dff; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: #1b1430;
    color: #ece6fa;
    border: 1px solid #3a2e63;
    selection-background-color: #7c4dff;
    outline: none;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid #5a4b8a;
    background-color: #160f29;
}
QCheckBox::indicator:checked { background-color: #7c4dff; border-color: #7c4dff; }
QRadioButton { spacing: 6px; background: transparent; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border-radius: 8px;
    border: 1px solid #5a4b8a;
    background-color: #160f29;
}
QRadioButton::indicator:checked { background-color: #7c4dff; border: 3px solid #160f29; }
QRadioButton:disabled { color: #6f6290; }
QListWidget {
    background-color: #160f29;
    border: 1px solid #2f2550;
    border-radius: 8px;
    padding: 4px;
    font-size: 12px;
}
QListWidget::item { padding: 3px 4px; border-bottom: 1px solid #221a3a; }
QListWidget::indicator {
    width: 15px; height: 15px;
    border-radius: 4px;
    border: 1px solid #5a4b8a;
    background-color: #160f29;
}
QListWidget::indicator:checked { background-color: #ff7a93; border-color: #ff7a93; }
QListWidget::item:selected { background-color: #3a2e63; color: white; }
QSlider::groove:horizontal {
    background: #160f29;
    height: 6px;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #7c4dff;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ece6fa;
    border: 3px solid #7c4dff;
    width: 12px;
    margin: -6px 0;
    border-radius: 9px;
}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #3a2e63;
    border-radius: 3px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover { background: #7c4dff; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { height: 0px; }

/* ---------- Tables (confusion matrix) ---------- */
QTableWidget {
    background-color: #160f29;
    gridline-color: #2f2550;
    border: 1px solid #2f2550;
    border-radius: 8px;
}
QHeaderView::section {
    background-color: #241a3d;
    color: #c9b3ff;
    border: none;
    padding: 4px;
    font-weight: bold;
}
"""
