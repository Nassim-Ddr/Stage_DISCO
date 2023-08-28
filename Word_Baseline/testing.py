import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QTextEdit,QInputDialog
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class WordReplacerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.text_edit = QTextEdit(self)
        self.replace_button = QPushButton("Highlight All", self)
        self.replace_button.clicked.connect(self.highlight_all)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.replace_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def highlight_all(self):
        search_text, ok = QInputDialog.getText(self, "Highlight All", "Enter text to highlight:")
        if ok:
            self.highlight_occurrences(search_text)

    def highlight_occurrences(self, search_text):
        format = QTextCharFormat()
        format.setBackground(QColor("yellow"))

        cursor = QTextCursor(self.text_edit.document())
        while not cursor.isNull():
            cursor = self.text_edit.document().find(search_text, cursor)

            if not cursor.isNull():
                cursor.mergeCharFormat(format)
                cursor.clearSelection()

    def occ_mots(self, search_text):
        # Replace this method with your implementation to count occurrences
        # of the search_text in the text_edit content.
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordReplacerApp()
    window.show()
    sys.exit(app.exec_())