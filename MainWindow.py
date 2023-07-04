import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Word Like app")

        self.editor = QTextEdit()

        self.setFixedSize(QSize(400, 300))

        # Set the central widget of the Window.
        self.setCentralWidget(self.editor)

    def keyPressEvent(self, event):
        print(event)
        print(QKeySequence.Copy)
        if event.matches(QKeySequence.Copy):
            self.handle_copy()
        elif event.matches(QKeySequence.Paste):
            self.handle_paste()
        else:
            super().keyPressEvent(event)

    def handle_copy(self):
        self.editor.copy()
        print("Copy shortcut activated")

    def handle_paste(self):
        self.editor.paste()
        print("Paste shortcut activated")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()