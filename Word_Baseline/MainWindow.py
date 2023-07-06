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
        elif event.key() == Qt.Key_U and modifiers == Qt.ControlModifier:
            self.handle_underline()
        elif event.key() == Qt.Key_B and modifiers == Qt.ControlModifier:
            self.handle_bold()
        elif event.key() == Qt.Key_Left:
            self.handle_move_cursor_left()
        elif event.key() == Qt.Key_Right:
            self.handle_move_cursor_right()

        # appel le comportement par defaut lie a l'evenement
        super().keyPressEvent(event)

    def handle_copy(self):
        # toPlainText() recupere le texte
        print(self.toPlainText())
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

        self.text_edit = CustomTextEdit(self)

        self.setFixedSize(QSize(400, 300))

        # Définit le widget central de la fenêtre.
        self.setCentralWidget(self.text_edit)

        search_action = QAction(QIcon(), "Search", self)
        search_action.setShortcut("Ctrl+F")
        search_action.triggered.connect(self.search)

        replace_action = QAction(QIcon(), "Replace", self)
        replace_action.setShortcut("Ctrl+R")
        replace_action.triggered.connect(self.replace)

        bold_action = QAction(QIcon(), "Bold", self)
        bold_action.setShortcut("Ctrl+B")
        bold_action.setCheckable(True)
        bold_action.toggled.connect(self.toggle_bold)

        underline_action = QAction(QIcon(), "Underline", self)
        underline_action.setShortcut("Ctrl+U")
        underline_action.setCheckable(True)
        underline_action.toggled.connect(self.toggle_underline)


        menu = self.menuBar()
        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction(search_action)
        edit_menu.addAction(replace_action)
        edit_menu.addSeparator()
        edit_menu.addAction(bold_action)
        edit_menu.addAction(underline_action)

    def search(self):
        # Demande à l'utilisateur d'entrer le texte à rechercher
        search_text, ok = QInputDialog.getText(self, "seach", "Enter text to search:")
        if ok:
            # Recherche la première occurrence du texte recherché dans le widget d'édition de texte
            cursor = self.text_edit.document().find(search_text)
            if not cursor.isNull():
                # Positionne le curseur de texte sur la position trouvée
                self.text_edit.setTextCursor(cursor)
                # Assure que le curseur est visible dans le widget d'édition de texte
                self.text_edit.ensureCursorVisible()
            else:
                # Informe l'utilisateur que le texte n'a pas été trouvé
                QMessageBox.information(self, "Search", "Text not found.")

    def replace(self):
        # Demande à l'utilisateur d'entrer le texte à remplacer
        search_text, ok = QInputDialog.getText(self, "Replace", "Enter text to replace:")
        if ok:
            # Demande à l'utilisateur d'entrer le texte de remplacement
            replace_text, ok = QInputDialog.getText(self, "Replace", "Enter replacement text:")
            if ok:
                document = self.text_edit.document()
                # Recherche la première occurrence du texte recherché dans le widget d'édition de texte
                cursor = document.find(search_text)
                while not cursor.isNull():
                    # Remplace le texte par le texte de remplacement
                    cursor.insertText(replace_text)
                    # Recherche l'occurrence suivante du texte recherché
                    cursor = document.find(search_text, cursor)

    def toggle_bold(self, boolgras):
        #Transforme le texte en gras (built in dans pyqt)
        font = self.text_edit.currentFont()
        font.setBold(boolgras)
        self.text_edit.setCurrentFont(font)

    def toggle_underline(self, boolsouligner):
        #Souligne le texte selectionnee 
        font = self.text_edit.currentFont()
        font.setUnderline(boolsouligner)
        self.text_edit.setCurrentFont(font)


    


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()