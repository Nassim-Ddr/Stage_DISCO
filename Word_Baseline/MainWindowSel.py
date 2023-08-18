import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from collections import Counter
from MapperLog2 import MapperLog2

from time import sleep
from PyQt5.QtTest import QTest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from Recommender import *
import random 
import copy

#CustomTextEdit, on en a besoin pour recuperer les commandes par defaut
class CustomTextEdit(QTextEdit):
    # customTextChanged = pyqtSignal(str)
    def __init__(self, parent= None,onWrite = False):
        super().__init__(parent)
        # isWrite = True -> on cree un csv contenant les donnees
        self.isWrite = onWrite
        self.logger = MapperLog2(onWrite)
        #self.setPlainText("Darkness. Just darkness. Darkness not visible. The absence of light. A vacuum I can’t describe. An . . . emptiness.\n\nDarkness in which I can see nothing. Darkness that terrifies me, suffocates me, crushes me. Darkness forced on me whether I like it or not, whether it is daylight or nighttime outside, in which I am expected to sleep. Darkness created by window coverings that cut off light and fresh air, the windows further curtained to prevent stray outside light from entering my room. Darkness.\n\nIn the darkness all I can hear is my clock. And my own heartbeat. And my breathing. At least I am alive. Or am I? It is hard to be sure in the darkness. Darkness, and voices. The voices of my parents, though I am alone, far from them: “Child do this . . . Child do that . . . Child don’t . . . Child why can’t you . . . Child stop . . . Child you must . . . Child, child, child.” Never, ever my name.\n\nLying there, I feel as if I am being forced into a pit, a hole in the ground—being buried, hidden, put away. As if I am disposable. As if my very existence is being denied. As if I must not be seen or heard. As if my birth is a dirty secret, an evil act of mine that must be obliterated without trace. As if I am an object of . . . shame. Why? What could I, a mere child, have done that would cause such a reaction in others, in my father and mother—the man and woman who created me, guardians and enforcers of my darkness?\n\nI am their only child. I think I know why: they never wanted me. I was an accident for them, a mistake they will be careful never to repeat.")
        self.setPlainText("The sun rises, and the morning brings a new day. Birds chirp, filling the air with melody.\nI wake up, feel refreshed, and get ready for the day ahead. I put on a comfortable shirt and jeans,\nready for a walk in the nearby park. As I stroll through the park, I see many people enjoying the outdoors.\nSome jog, others walk their dog. A child plays on the playground, laughing and having fun. It's a pleasant sight.\nIn the park, there is a tall tree providing shade from the sun. I find a bench under a tree and sit down to read a book.\nThe story is captivating, and time passes quickly. I lose myself in the pages, engrossed in the tale.\nAfter a while, I decide to explore more of the park. There is a pond with a duck swimming peacefully.\nI watch it for a while before continuing on my way. A group of friends has a picnic nearby,\nsharing food and laughter. As I walk, I notice a beautiful flower in various colors - red, blue, yellow, and white.\nThe fragrance of the flower fills the air,\nmaking the stroll even more delightful. I take a moment to admire the beauty of nature.\nAfter a pleasant walk, I head to a café for lunch. The café is cozy, and the menu offers a variety of dishes.\nI order a sandwich and a refreshing lemonade. The food is delicious, and I enjoy the peaceful atmosphere of the café.\nIn the afternoon, I meet a friend at the library. We browse through books, discussing our favorite author and genre.\nThe library is a treasure trove of knowledge, and we leave with a few books to read later.\nLater in the evening, I attend a music concert in the park. The band plays lively tunes,\nand the crowd claps along. The atmosphere is festive, and everyone enjoys the performance.\nAs the sun sets, the sky changes colors, displaying shades of orange and pink. It's a beautiful sight.\nIn conclusion, spending time outdoors and appreciating the simple pleasure of life can bring joy and fulfillment.\nThe NGSL provides a wide range of words to express the experience and emotion we encounter every day.")
        
        # On va placer le curseur au centre du document
        text_length = len(self.toPlainText())
        mid = text_length//2
        cursor = self.textCursor()
        cursor.setPosition(mid,QTextCursor.MoveAnchor)
        self.updated("")
        # if not onWrite :
        #     self.textChanged.connect(self.updated)
        #     self.customTextChanged.connect(self.keyPressEvent)


        # Fait en sorte que le cursor soit bien au milieu (le curseur est bien place)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    

    def keyPressEvent(self, event):
        # si on appuie sur CTRL
        modifiers = QApplication.keyboardModifiers()
        # appel le comportement par defaut lie a l'evenement

        # On block les commandes undo et redo pour l'experience
        blockUndo = (event.key() == Qt.Key_Z and modifiers == Qt.ControlModifier)
        blockRedo = (event.key() == Qt.Key_Y and modifiers == Qt.ControlModifier)

        cursor = self.textCursor()
        if (blockUndo or blockRedo):
            return
        super().keyPressEvent(event)

        keyValue = event.text()

        # Les differents comportements

        

        # supprimer un mot
        if event.key() == Qt.Key_Backspace and modifiers == Qt.ControlModifier:
            #print("Passing here")
            self.handle_backspace()
            return
        # souligne
        # elif event.key() == Qt.Key_U and modifiers == Qt.ControlModifier:
        #     self.handle_underline()
        # # met en gras
        # elif event.key() == Qt.Key_B and modifiers == Qt.ControlModifier:
        #     self.handle_bold()

        if event.key() == Qt.Key_V and modifiers == Qt.ControlModifier:
            self.handle_paste()
            return

        # Detection que lorsque le logiciel est en live 
        
        # if event.key() == Qt.Key_X and modifiers == Qt.ControlModifier:
        #     self.handle_cut()
        # if event.key() == Qt.Key_V and modifiers == Qt.ControlModifier:
        #     self.handle_paste()
        #     return
        # elif event.key() == Qt.Key_Z and modifiers == Qt.ControlModifier:
        #     self.handle_undo()
        # elif event.key() == Qt.Key_Y and modifiers == Qt.ControlModifier:
        #     self.handle_redo()

        # saut d'un mot vers la gauche/droite
        elif event.key() == Qt.Key_Left and modifiers == Qt.ControlModifier:
            self.handle_move_cursor_left()
            return
        elif event.key() == Qt.Key_Right and modifiers == Qt.ControlModifier:
            self.handle_move_cursor_right()
            return

        # selection de mot
        if (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier) and event.key() == Qt.Key_Left:
            self.handle_WordSelectionL()
            return
        elif (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier) and event.key() == Qt.Key_Right:
            self.handle_WordSelectionR()
            return
        
        # copier
        # elif event.key() == Qt.Key_C and modifiers == Qt.ControlModifier:
        #     self.handle_copy()
        

        
        

        # select line
        elif event.key() == Qt.Key_Home and modifiers == Qt.ShiftModifier:
            self.handle_selectLinetoStart()
            return
        elif event.key() == Qt.Key_End and modifiers == Qt.ShiftModifier:
            self.handle_selectLinetoEnd()
            return

        # saut de mot vers le debut ou fin d'une ligne
        elif event.key() == Qt.Key_Home:
            self.handle_move_start_line()
            return
        elif event.key() == Qt.Key_End:
            self.handle_move_end_line()
            return
        
        # select single character
        elif event.key() == Qt.Key_Right and modifiers == Qt.ShiftModifier:
            self.handle_selectionR()
            return
        elif event.key() == Qt.Key_Left and modifiers == Qt.ShiftModifier:
            self.handle_selectionL()
            return
        
        # selectionne tout le document
        elif event.key() == Qt.Key_A and modifiers == Qt.ControlModifier:
            self.handle_fullselection()
            return
        
        
        # On prend en compte les touches simples
        elif keyValue.isalpha() or keyValue.isdigit() or keyValue.isprintable() or event.key() == Qt.Key_Backspace :
            # ignorer ce cas 
            l = [16777238,16777235,16777227,16777237,16777239,16777222, 16777301, 16777239, 16777238, 16777264, 16777265, 16777266, 16777267, 16777330,16777328,16777329, 16777268,16777269,16777270,16777271,16777272,16777272,16777273,16777274,16777275, 16777249, 16777248,16777251, 16777250]

            #print(event.key())
            if  event.key() in l:
                # print("ça marche")
                return 
            self.handle_Letter()
            return
        
        
        

    # Handles for when not generating data -------------------------------------------------------------------------------
    
    def handle_copy(self):
        # toPlainText() recupere le texte
        #print(self.toPlainText())
        #print("J'ai copié")
        pass
    
    def handle_delete(self):
        print("Supprimer ?")
    
    # on groupe copier coller
    def handle_paste(self):
        #print("J'ai collé")
        self.updated("CopyPaste (CTRL + C -> CTRL + V)")

    def handle_cut(self):
        #print("J'ai coupé")
        pass

    def handle_undo(self):
        print("Annulay")

    def handle_redo(self):
        print("Annulay dans l'autre sens")

    def handle_move_cursor_left(self):
        #print(self.textCursor().position())
        #print("je vais a gauche ",self.textCursor().position())
        self.updated("CTRL + Left")

    def handle_move_cursor_right(self):
        #print(self.textCursor().position())
        #print("je vais à droite ",self.textCursor().position())
        self.updated("CTRL + Right") 

    def handle_move_start_line(self):
        #print("going to the start of the line")
        self.updated("Home")

    def handle_move_end_line(self):
        #print("going to the end of the line")
        self.updated("Fin (End)")
    
    def handle_selectionR(self):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        #self.updated("selectR")
        #self.updated()

    def handle_selectionL(self):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        #self.updated("selectL")
        #self.updated()
    

    def handle_WordSelectionR(self):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))

        #print("Test : ",cursor.blockNumber()," test 2 ",cursor.columnNumber())
        self.updated("CTRL + Shift + Right")

    def handle_WordSelectionL(self):
        cursor = self.textCursor()
        #print("Selection start: %d end: %d" % (cursor.selectionStart(), cursor.selectionEnd()))
        self.updated("CTRL + Shift + Left")
        

    def handle_fullselection(self):
        cursor = self.textCursor()
        #print("full selection")
        
        self.updated("CTRL+ A (SelectAll)")
        #print("after : ",cursor.blockNumber(),cursor.columnNumber())
    
    def handle_goLeftSingle(self):
        pass
        #self.updated()
    
    def handle_goRightSingle(self):
        pass
        #self.updated()
    
    def handle_selectLinetoStart(self):
        self.updated("Shift + Home")
    
    def handle_selectLinetoEnd(self):
        self.updated("Shift + End")

    
    # Others ---------------------------------------------------------------------

    def handle_backspace(self):
        #print("Supprimer le mot")
        self.updated("WordDel (CTRL + Backspace)")
    
    
    def handle_replace(self):
        self.updated("Search&Replace (CTRL + R)")
        #print("replace")
    
    
    def handle_tab(self):
        self.updated("tabbing")
    
    def handle_Letter(self):
        self.updated("WriteWord")


    # The update function that is called if command = None then we are online otherwise we are making a dataset
    def updated(self, command=None):
        cursor = self.textCursor()
        endPos = (cursor.blockNumber(),cursor.columnNumber())
        startSel = cursor.selectionStart()
        endSel = cursor.selectionEnd()
        charlen= len(self.toPlainText())

        text = self.toPlainText()
        word_count = len(text.split())
        self.logger.update(command, (self.toPlainText(),word_count,charlen,cursor.position(),endPos,startSel,endSel),self)
    


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
    def __init__(self,onWrite = False):
        super().__init__()

        self.setWindowTitle("Word Like app")

        self.text_edit = CustomTextEdit(self,onWrite = onWrite)
        self.text_edit.setMaximumWidth(750)

        #self.showMaximized()
        self.setMinimumSize(800,800)
        self.fontBox = QSpinBox()

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"Nombre de caractères : {len(self.text_edit.toPlainText())}")
        self.text_edit.textChanged.connect(lambda: self.statusBar.showMessage(f"Nombre de caractères : {len(self.text_edit.toPlainText())}"))
        self.actions = []

        # Définit le widget central de la fenêtre.
        central_widget = QWidget(self)
        
        layout = QHBoxLayout(central_widget)
        layout.addWidget(self.text_edit)



        self.word_frequency_widget = WordFrequencyWidget(self)
        layout.addWidget(self.text_edit)
        # layout.addWidget(self.word_frequency_widget)
        self.setCentralWidget(central_widget)
        layout.setContentsMargins(100,11,100,0)

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
        self.update_word_frequency()
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
    
    def replacePlayer(self,searchT,replaceT):
        # Demande à l'utilisateur d'entrer le texte à remplacer (Toujours QInputDialog)
        search_text, ok = searchT, True
        if ok:
            # Demande à l'utilisateur d'entrer le texte de remplacement 
            replace_text, ok = replaceT, True
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
        #print(search_text)
        self.text_edit.handle_replace()
    
    # Comment : It doesn't matter where we paste it as long as it's on an empty space due to how bag of word works
    def playerCopyPaste(self):
        texteditor = self.text_edit
        dice = np.random.randint(2,7)
        direction = (np.random.random() < 0.5)
        oldCursor = texteditor.textCursor()
        end = len(texteditor.toPlainText())

        # We force direction if at end of start of document because makes no sense otherwise
        if oldCursor.position() == end:
            direction = False
        elif oldCursor.position() == 0:
            direction = True

        # direction selected (left of right)
        if direction :
            # Number of element selected
            for i in range(dice):
                QTest.keyPress(texteditor, Qt.Key_Right, Qt.ControlModifier | Qt.ShiftModifier)
        # copy
        else : 
            # Number of element selected
            for i in range(dice):
                QTest.keyPress(texteditor, Qt.Key_Left, Qt.ControlModifier | Qt.ShiftModifier)

        QTest.keyPress(texteditor,Qt.Key_C,  Qt.ControlModifier)

        # move the cursor to the end of the document
        cursor = texteditor.textCursor()
        cursor.movePosition(cursor.End)
        texteditor.setTextCursor(cursor)
        QTest.keyPress(texteditor,Qt.Key_Space)

        # Pasting the text
        QTest.keyPress(texteditor,Qt.Key_V,  Qt.ControlModifier)

        texteditor.setTextCursor(oldCursor)

    
    def writeWord(self):
        randomletters = [random.choice('abcdefghijklmnopqrstuvwxyz') for i in range(np.random.randint(1,11))]
        texteditor = self.text_edit
        oldCursor = texteditor.textCursor()
        #oldPos = oldCursor.position()


        # move the cursor to the end of the document
        cursor = texteditor.textCursor()
        cursor.movePosition(cursor.End)
        texteditor.setTextCursor(cursor)
        QTest.keyPress(texteditor,Qt.Key_Space)

        for char in randomletters:
            QTest.keyClick(self.text_edit, char)
            QTest.qWait(2)  # Wait for a short duration between each character
            self.text_edit.updated("WriteWord")

        #print(f'old pos {oldCursor.position()} new pos {cursor.position()}')
        

        texteditor.setTextCursor(oldCursor)


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
    texteditor.setPlainText("The sun rises, and the morning brings a new day. Birds chirp, filling the air with melody.\nI wake up, feel refreshed, and get ready for the day ahead. I put on a comfortable shirt and jeans,\nready for a walk in the nearby park. As I stroll through the park, I see many people enjoying the outdoors.\nSome jog, others walk their dog. A child plays on the playground, laughing and having fun. It's a pleasant sight.\nIn the park, there is a tall tree providing shade from the sun. I find a bench under a tree and sit down to read a book.\nThe story is captivating, and time passes quickly. I lose myself in the pages, engrossed in the tale.\nAfter a while, I decide to explore more of the park. There is a pond with a duck swimming peacefully.\nI watch it for a while before continuing on my way. A group of friends has a picnic nearby,\nsharing food and laughter. As I walk, I notice a beautiful flower in various colors - red, blue, yellow, and white.\nThe fragrance of the flower fills the air,\nmaking the stroll even more delightful. I take a moment to admire the beauty of nature.\nAfter a pleasant walk, I head to a café for lunch. The café is cozy, and the menu offers a variety of dishes.\nI order a sandwich and a refreshing lemonade. The food is delicious, and I enjoy the peaceful atmosphere of the café.\nIn the afternoon, I meet a friend at the library. We browse through books, discussing our favorite author and genre.\nThe library is a treasure trove of knowledge, and we leave with a few books to read later.\nLater in the evening, I attend a music concert in the park. The band plays lively tunes,\nand the crowd claps along. The atmosphere is festive, and everyone enjoys the performance.\nAs the sun sets, the sky changes colors, displaying shades of orange and pink. It's a beautiful sight.\nIn conclusion, spending time outdoors and appreciating the simple pleasure of life can bring joy and fulfillment.\nThe NGSL provides a wide range of words to express the experience and emotion we encounter every day.")
    #texteditor.setPlainText("Darkness. Just darkness. Darkness not visible. The absence of light. A vacuum I can’t describe. An . . . emptiness.\n\nDarkness in which I can see nothing. Darkness that terrifies me, suffocates me, crushes me. Darkness forced on me whether I like it or not, whether it is daylight or nighttime outside, in which I am expected to sleep. Darkness created by window coverings that cut off light and fresh air, the windows further curtained to prevent stray outside light from entering my room. Darkness.\n\nIn the darkness all I can hear is my clock. And my own heartbeat. And my breathing. At least I am alive. Or am I? It is hard to be sure in the darkness. Darkness, and voices. The voices of my parents, though I am alone, far from them: “Child do this . . . Child do that . . . Child don’t . . . Child why can’t you . . . Child stop . . . Child you must . . . Child, child, child.” Never, ever my name.\n\nLying there, I feel as if I am being forced into a pit, a hole in the ground—being buried, hidden, put away. As if I am disposable. As if my very existence is being denied. As if I must not be seen or heard. As if my birth is a dirty secret, an evil act of mine that must be obliterated without trace. As if I am an object of . . . shame. Why? What could I, a mere child, have done that would cause such a reaction in others, in my father and mother—the man and woman who created me, guardians and enforcers of my darkness?\n\nI am their only child. I think I know why: they never wanted me. I was an accident for them, a mistake they will be careful never to repeat.")
    text_length = len(texteditor.toPlainText())

    position = np.random.randint(1,text_length)

    cursor = texteditor.textCursor()
    cursor.setPosition(position,QTextCursor.MoveAnchor)

    texteditor.setTextCursor(cursor)
    texteditor.ensureCursorVisible()
    texteditor.logger.reset()
    texteditor.updated("")


def get_dict():
    my_data = pd.read_csv('data/NGSLWords.csv', delimiter=',', usecols=[0]).to_numpy()[:,0]
    return my_data

def useAct(action,app,window):
    dico = get_dict()
    print("choice ",action)
    match action:
        case "SelectWR":
            QTest.keyPress(app, Qt.Key_Right, Qt.ControlModifier | Qt.ShiftModifier)
        case "SelectWL":
            QTest.keyPress(app, Qt.Key_Left, Qt.ControlModifier | Qt.ShiftModifier)
        case "SelectHome":
            QTest.keyPress(app, Qt.Key_Home,  Qt.ShiftModifier)
        case "SelectEnd":
            QTest.keyPress(app, Qt.Key_End,  Qt.ShiftModifier)
        case "MoveWR":
            QTest.keyPress(app, Qt.Key_Right,  Qt.ControlModifier)
        case "MoveWL":
            QTest.keyPress(app, Qt.Key_Left,  Qt.ControlModifier)
        case "MoveHome":
            QTest.keyPress(app, Qt.Key_Home)
        case "MoveEnd":
            QTest.keyPress(app, Qt.Key_End)
        case "SelectAll":
            QTest.keyPress(app,Qt.Key_A,  Qt.ControlModifier)
        case "WordDel":
            QTest.keyPress(app,Qt.Key_Backspace, Qt.ControlModifier)
        case "Replace":
            text = app.toPlainText()
            vectorizer = CountVectorizer()

            # Fit transform
            vectorizer.fit_transform([text])

            # Get names
            replaceable = vectorizer.get_feature_names_out()
            
            

            # the word to replace
            toreplace = np.random.choice(replaceable)
            dico = np.delete(dico,np.where(dico == toreplace))

            # the replacing word
            replaced = np.random.choice(dico)

            window.replacePlayer(toreplace,replaced)

        case "CopyPaste":
            window.playerCopyPaste()
        
        case "WriteWord" :
            window.writeWord()


    cursor = app.textCursor()
    cursor.clearSelection()
    app.setTextCursor(cursor)


def play(texteditor,window,commands,start_funtion=lambda: print(None), reset_function=lambda: print(None),epochs = 4000,moves =10):
    for i in range(epochs):
        for j in range(moves):
            cursor = texteditor.textCursor()
            act = np.random.choice(commands)
            if cursor.blockNumber() == 0 and cursor.columnNumber == 0:
                reset(texteditor)
            useAct(act,texteditor,window)
        reset(texteditor)


    

# Machine learning data creation that does not work well in practice for selection
"""def play(texteditor,window,commandList1,commandList2,commandList3,start_funtion=lambda: print(None), reset_function=lambda: print(None),epochs = 100,moves = 5):

    tmpep = epochs*len(commandList3)
    tmpmoves = moves*len(commandList3)
    for i in range(tmpep):
        for j in range(tmpmoves):
            act = np.random.choice(commandList3)
            useAct(act,texteditor,window)
        reset(texteditor)
    

    tmpep = epochs*len(commandList1)
    tmpmoves = moves*len(commandList1)

    for i in range(tmpep):
        for j in range(tmpmoves):
            act = np.random.choice(commandList1)
            useAct(act,texteditor,window)
        reset(texteditor)
    

    tmpep = epochs*len(commandList2)
    tmpmoves = moves*len(commandList2)

    for i in range(tmpep):
        for j in range(tmpmoves):
            act = np.random.choice(commandList2)
            useAct(act,texteditor,window)
        reset(texteditor)"""
    

"""def playNonRand(texteditor,window,commandList1,commandList2,commandList3,start_funtion=lambda: print(None), reset_function=lambda: print(None),epochs = 500,moves = 5):

    for k in range(len(commandList3)):
        act = commandList3[k]
        for i in range(epochs):
            for j in range(moves):
                useAct(act,texteditor,window)
            reset(texteditor)
    
    for k in range(len(commandList1)):
        act = commandList1[k]
        for i in range(epochs):
            for j in range(moves):
                useAct(act,texteditor,window)
            reset(texteditor)
    

    for k in range(len(commandList2)):
        act = commandList2[k]
        for i in range(epochs):
            for j in range(moves):
                useAct(act,texteditor,window)
            reset(texteditor)"""
    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow(onWrite = True)
    window.show()
    #R = Recommender("./models/bowModelGood",window.text_edit,hardCoded = True)
    #window.text_edit.logger.assistant = R
    #R.show()
    #actions = ["SelectWR","SelectWL","SelectShiftR","SelectShiftL","MoveWR","MoveWL","MoveHome","MoveEnd","Tab","SelectAll"]
    #actionsSel = ["SelectWR","SelectWL","SelectAll"]
    #actionsMove = ["MoveWR","MoveWL","MoveHome","MoveEnd","Tab"]
    # Copy paste bigram
    #actionsWord = ["WordDel","Replace","CopyPaste","WriteWord","SelectWR","SelectWL","MoveWR","MoveWL","MoveHome","MoveEnd","SelectAll","SelectHome","SelectEnd"]
    actions = ["MoveWR","MoveWL","MoveHome","MoveEnd","SelectWR","SelectWL","SelectHome","SelectEnd","SelectAll"]
    #play(window.text_edit,window,actionsSel,actionsMove,actionsWord,reset_function=reset)
    #playNonRand(window.text_edit,window,actionsSel,actionsMove,actionsWord,reset_function=reset)
    play(window.text_edit,window,actions)

    app.exec()

    window.text_edit.logger.file2.close()
    window.text_edit.logger.file.close()
    