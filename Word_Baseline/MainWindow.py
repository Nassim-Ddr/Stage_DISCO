import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import signal


#CustomTextEdit, on en a besoin pour recuperer les commandes par defaut
class CustomTextEdit(QTextEdit):
    def keyPressEvent(self, event):

        # si on appuie sur CTRL
        modifiers = QApplication.keyboardModifiers()

        # Les differents comportements
        if event.key() == Qt.Key_Backspace:
            self.handle_backspace()
        elif event.key() == Qt.Key_Delete:
            self.handle_delete()
        elif event.key() == Qt.Key_V and modifiers == Qt.ControlModifier:
            self.handle_paste()
        elif event.key() == Qt.Key_X and modifiers == Qt.ControlModifier:
            self.handle_cut()
        elif event.key() == Qt.Key_Z and modifiers == Qt.ControlModifier:
            self.handle_undo()
        elif event.key() == Qt.Key_Y and modifiers == Qt.ControlModifier:
            self.handle_redo()
        elif event.key() == Qt.Key_C and modifiers == Qt.ControlModifier:
            self.handle_copy()
        elif event.key() == Qt.Key_Left:
            self.handle_move_cursor_left()
        elif event.key() == Qt.Key_Right:
            self.handle_move_cursor_right()

        # appel le comportement par defaut lie a l'evenement
        super().keyPressEvent(event)

    def handle_copy(self):
        print("J'ai copié")

    def handle_backspace(self):
        print("Supprimer le caractere")

    def handle_delete(self):
        print("Supprimer ?")

    def handle_paste(self):
        print("J'ai collé")

    def handle_cut(self):
        print("J'ai coupé")

    def handle_undo(self):
        print("Annulay")

    def handle_redo(self):
        print("Annulay dans l'autre sens")

    def handle_move_cursor_left(self):
        print("je vais a gauche")

    def handle_move_cursor_right(self):
        print("je vais à droite")

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Word Like app")

        self.editor = CustomTextEdit()


        self.setFixedSize(QSize(400, 300))

        # Set the central widget of the Window.
        self.setCentralWidget(self.editor)

    


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()