from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys


class MainWindow(QWidget):
    def __init__(self,slides):
        super().__init__()
        self.slides = slides
        self.curslide = 0
        self.init_ui()
        self.setFixedSize(800,800)

    def init_ui(self):
        self.setWindowTitle("PowerPoint like")
        self.layout = QVBoxLayout()
        

        self.slide_label = QLabel()
        self.slide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.slide_label)


        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_slide)
        self.layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_slide)
        self.layout.addWidget(self.next_button)


        self.setLayout(self.layout)
        self.show_slide()


    def show_slide(self):
        self.slide_label.setText(self.slides[self.curslide])
    
    def show_next_slide(self):
        if self.curslide <len(self.slides)-1:
            self.curslide+=1
            self.show_slide()
    
    def show_previous_slide(self):
        if self.curslide > 0:
            self.curslide-=1
            self.show_slide()



if __name__ == '__main__':
    app = QApplication(sys.argv)


    slides = ["test1","test2","test3"]
    
    window = MainWindow(slides)

    window.show()

    sys.exit(app.exec_())