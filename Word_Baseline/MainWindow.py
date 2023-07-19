import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import signal
from collections import Counter
from MapperLog import MapperLog
from MapperLog2 import MapperLog2

from threading import Thread
import pyautogui
from time import sleep
from PyQt5.QtTest import QTest
import numpy as np


#CustomTextEdit, on en a besoin pour recuperer les commandes par defaut
class CustomTextEdit(QTextEdit):
    def __init__(self, parent= None):
        super().__init__(parent)
        self.logger = MapperLog2()
        self.setPlainText("Darkness. Just darkness. Darkness not visible. The absence of light. A vacuum I can’t describe. An . . . emptiness.\n\nDarkness in which I can see nothing. Darkness that terrifies me, suffocates me, crushes me. Darkness forced on me whether I like it or not, whether it is daylight or nighttime outside, in which I am expected to sleep. Darkness created by window coverings that cut off light and fresh air, the windows further curtained to prevent stray outside light from entering my room. Darkness.\n\nIn the darkness all I can hear is my clock. And my own heartbeat. And my breathing. At least I am alive. Or am I? It is hard to be sure in the darkness. Darkness, and voices. The voices of my parents, though I am alone, far from them: “Child do this . . . Child do that . . . Child don’t . . . Child why can’t you . . . Child stop . . . Child you must . . . Child, child, child.” Never, ever my name.\n\nLying there, I feel as if I am being forced into a pit, a hole in the ground—being buried, hidden, put away. As if I am disposable. As if my very existence is being denied. As if I must not be seen or heard. As if my birth is a dirty secret, an evil act of mine that must be obliterated without trace. As if I am an object of . . . shame. Why? What could I, a mere child, have done that would cause such a reaction in others, in my father and mother—the man and woman who created me, guardians and enforcers of my darkness?\n\nI am their only child. I think I know why: they never wanted me. I was an accident for them, a mistake they will be careful never to repeat.")
        text_length = len(self.toPlainText())
        mid = text_length//2
        cursor = self.textCursor()
        cursor.setPosition(mid,QTextCursor.MoveAnchor)

        # Fait en sorte que le cursor soit bien au milieu
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def keyPressEvent(self, event):
        # si on appuie sur CTRL
        modifiers = QApplication.keyboardModifiers()
        # appel le comportement par defaut lie a l'evenement
        cursor = self.textCursor()
        starting = (cursor.blockNumber(),cursor.columnNumber())
        super().keyPressEvent(event)

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
        elif event.key() == Qt.Key_Left and modifiers == Qt.ControlModifier:
            self.handle_move_cursor_left()
        elif event.key() == Qt.Key_Right and modifiers == Qt.ControlModifier:
            self.handle_move_cursor_right()
        elif event.key() == Qt.Key_Home:
            self.handle_move_start_line()
        elif event.key() == Qt.Key_End:
            self.handle_move_end_line()
        elif event.key() == Qt.Key_Right and modifiers == Qt.ShiftModifier:
            self.handle_selectionR(starting)
        elif event.key() == Qt.Key_Left and modifiers == Qt.ShiftModifier:
            self.handle_selectionL(starting)
        elif (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier) and event.key() == Qt.Key_Left:
            self.handle_WordSelectionR(starting)
        elif (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier) and event.key() == Qt.Key_Right:
            self.handle_WordSelectionR(starting)
        elif event.key() == Qt.Key_A and modifiers == Qt.ControlModifier:
            self.handle_fullselection(starting)
        elif event.key() == Qt.Key_Tab :
            self.handle_tab()

        

    def handle_copy(self):
        # toPlainText() recupere le texte
        #print(self.toPlainText())
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
        #print(self.textCursor().position())
        print("je vais a gauche")
        self.updated("moveL")

    def handle_move_cursor_right(self):
        #print(self.textCursor().position())
        print("je vais à droite")
        self.updated("moveR")
    
    def handle_replace(self):
        self.updated("replace")
        print("replace")

    def updated(self, command,startingPos = (0,0),selection = False):
        # Si ce n'est pas un commande de selection, on suppose que c'est nulle
        cursor = self.textCursor()
        if not (selection) :
            endPos = startingPos
        else :
            
            endPos = (cursor.blockNumber(),cursor.columnNumber())
        self.logger.update(command, (self.toPlainText(),cursor.position(),startingPos,endPos))
    
    def handle_move_start_line(self):
        print("going to the start of the line")

    def handle_move_end_line(self):
        print("going to the end of the line")
    
    def handle_selectionR(self,startingPos):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        self.updated("selectR",startingPos,True)

    def handle_selectionL(self,startingPos):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        self.updated("selectL",startingPos,True)
    

    def handle_WordSelectionR(self,startingPos):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))

        print("Test : ",cursor.blockNumber()," test 2 ",cursor.columnNumber())
        self.updated("selectWR",startingPos,True)

    def handle_WordSelectionL(self,startingPos):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        self.updated("selectLR",startingPos,True)
        

    def handle_fullselection(self,startingPos):
        cursor = self.textCursor()
        self.updated("selectAll",startingPos,True)
        print("full selection",startingPos)
    
    def handle_tab(self):
        self.updated("tabbing")


class WordFrequencyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout()
        self.label = QLabel(self)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def update_word_frequency(self, text):
        word_count = Counter(text.split())
        self.label.setText("Word Frequency:\n" + "\n".join(f"{word}: {count}" for word, count in word_count.items()))
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Word Like app")

        self.text_edit = CustomTextEdit(self)

        #self.showMaximized()
        self.setFixedSize(800,800)
        self.fontBox = QSpinBox()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.text_edit.textChanged.connect(lambda: self.statusBar.showMessage(f"Nombre de caractères : {len(self.text_edit.toPlainText())}"))
        self.actions = []

        # Définit le widget central de la fenêtre.
        central_widget = QWidget(self)
        layout = QHBoxLayout(central_widget)
        layout.addWidget(self.text_edit)

        self.word_frequency_widget = WordFrequencyWidget(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.word_frequency_widget)
        self.setCentralWidget(central_widget)

        search_action = QAction(QIcon(), "Search", self)
        search_action.setShortcut("Ctrl+F")
        search_action.triggered.connect(self.search)

        replace_action = QAction(QIcon(), "Replace", self)
        replace_action.setShortcut("Ctrl+R")
        replace_action.triggered.connect(self.replace)

        bold_action = QAction(QIcon("./icons/bold.png"), "Bold", self)
        bold_action.setShortcut("Ctrl+B")
        #Permet de detecter qu'un bouton est utilisé
        bold_action.setCheckable(True)
        # slot toggle_bold plus bas
        bold_action.toggled.connect(self.toggle_bold)

        underline_action = QAction(QIcon("./icons/underlined.png"), "Underline", self)
        underline_action.setShortcut("Ctrl+U")
        underline_action.setCheckable(True)
        # slot toggle_underline plus bas
        underline_action.toggled.connect(self.toggle_underline)

        italic_action = QAction(QIcon("./icons/italic.png"), "Italic", self)
        italic_action.setShortcut("Ctrl+I")
        italic_action.setCheckable(True)
        # slot toggle_underline plus bas
        italic_action.toggled.connect(self.toggle_italic)


        menu = self.menuBar()
        edit_menu = menu.addMenu("&Operation")
        edit_menu.addAction(search_action)
        edit_menu.addAction(replace_action)
        edit_menu.addSeparator()
        edit_menu.addAction(bold_action)
        edit_menu.addAction(underline_action)
        edit_menu.addAction(italic_action)


        # On ajoute une toolbar pour mettre en gras et souligner
        toolbar = QToolBar()
        actUndo = QAction(QIcon("./icons/undo.png"), "Undo", self)
        actUndo.triggered.connect(self.text_edit.undo)
        toolbar.addAction(actUndo)

        actRedo = QAction(QIcon("./icons/redo.png"), "Redo", self)
        actRedo.triggered.connect(self.text_edit.redo)
        toolbar.addAction(actRedo)

        toolbar.addSeparator()

        toolbar.addAction(bold_action)
        toolbar.addAction(underline_action)
        toolbar.addAction(italic_action)

        toolbar.addSeparator()

        # Actions d'alignements
        alignL = QAction(QIcon("./icons/alignLeft.png"), "Left Allign", self)
        alignL.triggered.connect(lambda : self.text_edit.setAlignment(Qt.AlignLeft))
        toolbar.addAction(alignL)    

        alignMid = QAction(QIcon("./icons/alignCent.png"), "Center Allign", self)
        alignMid.triggered.connect(lambda : self.text_edit.setAlignment(Qt.AlignCenter))
        toolbar.addAction(alignMid)

        alignR = QAction(QIcon("./icons/alignRight.png"), "Right Allign", self)
        alignR.triggered.connect(lambda : self.text_edit.setAlignment(Qt.AlignRight))
        toolbar.addAction(alignR)

        self.fontTb = QComboBox(self)
        self.fontTb.addItems(["Arial","Courier Std","Monospace"])
        self.fontTb.activated.connect(self.setFont)
        toolbar.addWidget(self.fontTb)
        self.text_edit.setCurrentFont(QFont("Arial"))
        self.text_edit.setFontPointSize(24)

        self.addToolBar(toolbar)

        # Change la taille du texte
        self.fontBox.setValue(24)
        self.fontBox.valueChanged.connect(self.setFontSize)
        toolbar.addWidget(self.fontBox)

        # Connect the text changed signal to update the word frequency
        self.text_edit.textChanged.connect(self.update_word_frequency)

    def update_word_frequency(self):
        text = self.text_edit.toPlainText()
        self.word_frequency_widget.update_word_frequency(text)
        

    def search(self):
        # Demande à l'utilisateur d'entrer le texte à rechercher (QInputDialog)
        search_text, ok = QInputDialog.getText(self, "seach", "Enter text to search:")
        if ok:
            # Cherche la première occurrence du texte recherché dans le widget d'édition de texte
            cursor = self.text_edit.document().find(search_text)
            if not cursor.isNull():
                # Positionne le curseur de texte sur la position
                self.text_edit.setTextCursor(cursor)
                # Assure que le curseur est visible dans le widget d'editeur de texte
                self.text_edit.ensureCursorVisible()
            else:
                # Informe l'utilisateur que le texte n'a pas été trouvé
                QMessageBox.information(self, "Search", "Text not found.")

    def replace(self):
        # Demande à l'utilisateur d'entrer le texte à remplacer (Toujours QInputDialog)
        search_text, ok = QInputDialog.getText(self, "Replace", "Enter text to replace:")
        if ok:
            # Demande à l'utilisateur d'entrer le texte de remplacement 
            replace_text, ok = QInputDialog.getText(self, "Replace", "Enter replacement text:")
            if ok:
                document = self.text_edit.document()
                # Recherche la première occurrence du texte recherché dans le widget d'édition de texte
                # Toutes les occurrences seront remplacees
                cursor = document.find(search_text)
                while not cursor.isNull():
                    # Remplace l'élément par le nouveau texte
                    cursor.insertText(replace_text)
                    # Recherche l'occurrence suivante du texte recherché
                    cursor = document.find(search_text, cursor)
        self.text_edit.handle_replace()


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

    def toggle_italic(self, boolita):
        #Souligne le texte selectionnee 
        font = self.text_edit.currentFont()
        font.setItalic(boolita)
        self.text_edit.setCurrentFont(font)

    
    def setFont(self):
        font = self.text_edit.currentFont()
        font.setFamily(self.fontTb.currentText())
        self.text_edit.setCurrentFont(font)

    
    def setFontSize(self):
        val = self.fontBox.value()
        self.text_edit.setFontPointSize(val)
        
    


def reset(texteditor):

    texteditor.clear()
    texteditor.setPlainText("Darkness. Just darkness. Darkness not visible. The absence of light. A vacuum I can’t describe. An . . . emptiness.\n\nDarkness in which I can see nothing. Darkness that terrifies me, suffocates me, crushes me. Darkness forced on me whether I like it or not, whether it is daylight or nighttime outside, in which I am expected to sleep. Darkness created by window coverings that cut off light and fresh air, the windows further curtained to prevent stray outside light from entering my room. Darkness.\n\nIn the darkness all I can hear is my clock. And my own heartbeat. And my breathing. At least I am alive. Or am I? It is hard to be sure in the darkness. Darkness, and voices. The voices of my parents, though I am alone, far from them: “Child do this . . . Child do that . . . Child don’t . . . Child why can’t you . . . Child stop . . . Child you must . . . Child, child, child.” Never, ever my name.\n\nLying there, I feel as if I am being forced into a pit, a hole in the ground—being buried, hidden, put away. As if I am disposable. As if my very existence is being denied. As if I must not be seen or heard. As if my birth is a dirty secret, an evil act of mine that must be obliterated without trace. As if I am an object of . . . shame. Why? What could I, a mere child, have done that would cause such a reaction in others, in my father and mother—the man and woman who created me, guardians and enforcers of my darkness?\n\nI am their only child. I think I know why: they never wanted me. I was an accident for them, a mistake they will be careful never to repeat.")
    text_length = len(texteditor.toPlainText())

    mid = text_length//2

    cursor = texteditor.textCursor()
    cursor.setPosition(mid,QTextCursor.MoveAnchor)

    texteditor.setTextCursor(cursor)
    texteditor.ensureCursorVisible()

def useAct(action,app):
    match action:
        case "SelectWR":
            QTest.keyPress(app, Qt.Key_Right, Qt.ControlModifier | Qt.ShiftModifier)
        case "SelectWL":
            QTest.keyPress(app, Qt.Key_Left, Qt.ControlModifier | Qt.ShiftModifier)
        case "SelectShiftR":
            QTest.keyPress(app, Qt.Key_Right,  Qt.ShiftModifier)
        case "SelectShiftL":
            QTest.keyPress(app, Qt.Key_Left,  Qt.ShiftModifier)
        case "MoveWR":
            QTest.keyPress(app, Qt.Key_Right,  Qt.ControlModifier)
        case "MoveWL":
            QTest.keyPress(app, Qt.Key_Left,  Qt.ControlModifier)
        case "MoveHome":
            QTest.keyPress(app, Qt.Key_Home)
        case "MoveEnd":
            QTest.keyPress(app, Qt.Key_End)
        case "Tab":
            QTest.keyPress(app,Qt.Key_Tab)
        case "SelectAll":
            QTest.keyPress(app,Qt.Key_A,  Qt.ControlModifier)
        

def play(texteditor,commandList,start_funtion=lambda: print(None), reset_function=lambda: print(None),epochs = 1000,moves = 3):

    for i in range(epochs):
        for j in range(moves):
            act = np.random.choice(actions)
            useAct(act,texteditor)
        reset(texteditor)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    actions = ["SelectWR","SelectWL","SelectShiftR","SelectShiftL","MoveWR","MoveWL","MoveHome","MoveEnd","Tab","SelectAll"]

    play(window.text_edit,actions,reset_function=reset)

    app.exec()

    window.text_edit.logger.file2.close()
    #player = WordPlayer()
    #self.player.app.text_edit.logger.file2.close()
    #self.player.app.text_edit.logger.file.close()
    
