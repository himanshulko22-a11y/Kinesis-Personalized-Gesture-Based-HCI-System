"""Dark purple theme for the Kinesis window."""

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
QLabel#note {
    color: #9d8ec2;
    font-size: 11px;
}
"""
