import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from MainWindow import *

class TitleBar(QDialog):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        css = """
        QWidget{
            Background: #2B579A;
            color:white;
            font:12px bold;
            font-weight:bold;
            border-radius: 1px;
            height: 11px;
        }
        QDialog{
            font-size:12px;
            color: black;

        }
        QToolButton{
            Background:#2B579A;
            font-size:11px;
        }
        QToolButton:hover{
            Background: #dc5939;
            font-size:11px;
        }
        """
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.Highlight)
        self.setStyleSheet(css) 
        self.minimize=QToolButton(self)
        self.minimize.setIcon(QIcon("icons/remove.png"))
        self.maximize=QToolButton(self)
        self.maximize.setIcon(QIcon("icons/resize.png"))
        close=QToolButton(self)
        close.setIcon(QIcon("icons/close.png"))
        self.minimize.setMinimumHeight(20)
        close.setMinimumHeight(20)
        self.maximize.setMinimumHeight(20)
        label=QLabel(self)
        label.setText("Word Like App")
        self.setWindowTitle("Word Like App")
        hbox=QHBoxLayout(self)
        hbox.addWidget(label)
        hbox.addWidget(self.minimize)
        hbox.addWidget(self.maximize)
        hbox.addWidget(close)
        hbox.insertStretch(1,500)
        hbox.setSpacing(0)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.maxNormal=False
        close.clicked.connect(self.close)
        self.minimize.clicked.connect(self.showSmall)
        self.maximize.clicked.connect(self.showMaxRestore)

    def showSmall(self):
        box.showMinimized()

    def showMaxRestore(self):
        if(self.maxNormal):
            box.showNormal()
            self.maxNormal= False
            self.maximize.setIcon(QIcon("icons/resize.png"))
            print('1')
        else:
            box.showMaximized()
            self.maxNormal=  True
            print('2')
            self.maximize.setIcon(QIcon("icons/resize.png"))

    def close(self):
        QCoreApplication.quit()
        self.parent().close()


    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            box.moving = True 
            box.offset = event.pos()

    def mouseMoveEvent(self,event):
        if box.moving: 
            box.move(event.globalPos()-box.offset)


class Frame(QFrame):
    def __init__(self, parent=None):
        QFrame.__init__(self, parent)
        self.m_mouse_down= False
        self.setFrameShape(QFrame.StyledPanel)
        css = """
        """
        self.setStyleSheet(css) 
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setMouseTracking(True)
        self.m_titleBar= TitleBar(self)
        self.m_content= QWidget(self)
        vbox=QVBoxLayout(self)
        vbox.addWidget(self.m_titleBar)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(0)
        layout=QVBoxLayout(self)
        layout.addWidget(self.m_content)
        #layout.setContentsMargins(5,5,5,5)
        layout.setSpacing(0)
        vbox.addLayout(layout)
        
        # Allows you to access the content area of the frame
        # where widgets and layouts can be added
        window = MainWindow(onWrite = False)
        
        R = Recommender("./models/bowModelGood",window.text_edit,hardCoded = True)
        window.text_edit.logger.assistant = R
        R.show()
        vbox.addWidget(window)


    def contentWidget(self):
        return self.m_content

    def titleBar(self):
        return self.m_titleBar

    def mousePressEvent(self,event):
        self.m_old_pos = event.pos()
        self.m_mouse_down = event.button()== Qt.LeftButton

    def mouseMoveEvent(self,event):
        x=event.x()
        y=event.y()

    def mouseReleaseEvent(self,event):
        m_mouse_down=False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    box = Frame()
    box.move(60,60)
    l=QVBoxLayout(box.contentWidget())
    l.setContentsMargins(0,0,0,0)
    box.show()
    app.exec_()