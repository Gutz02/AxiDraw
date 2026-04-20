import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt

class TabletWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tablet Test")
        self.resize(600, 400)

    def tabletEvent(self, event):
        pressure = event.pressure()
        button = event.button()
        buttons = event.buttons()
        pos = event.position()

        tip_down = pressure > 0.01

        print(
            f"x={pos.x():.1f}, y={pos.y():.1f}, \n"
            f"pressure={pressure:.3f}, \n"
            f"button={button}, buttons={buttons}, \n"
            f"tip_down={tip_down}\n"
        )

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TabletWidget()
    w.show()
    sys.exit(app.exec())