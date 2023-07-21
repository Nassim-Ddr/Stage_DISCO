# importing the required libraries

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


class CustomWindow(QMainWindow):
    def paintEvent(self, event=None):
        painter = QPainter(self)

        painter.setOpacity(0.7)
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.white))   
        painter.drawRect(self.rect())


app = QApplication(sys.argv)

# Create the main window
window = CustomWindow()

window.setWindowFlags(Qt.FramelessWindowHint)
window.setAttribute(Qt.WA_NoSystemBackground, True)
window.setAttribute(Qt.WA_TranslucentBackground, True)

# Create the button
pushButton = QPushButton(window)
pushButton.setGeometry(QRect(240, 190, 90, 31))
pushButton.setText("Finished")
pushButton.clicked.connect(app.quit)

# Center the button
qr = pushButton.frameGeometry()
cp = QDesktopWidget().availableGeometry().center()
qr.moveCenter(cp)
pushButton.move(qr.topLeft())

# Run the application
window.show()
sys.exit(app.exec_())