import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QColorDialog, QToolBar, QDialog, QSizePolicy, QToolButton, QAction
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QWidget,  QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QKeySequence, QColor, QPalette, QIcon, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

from PyQt5 import QtGui
from PyQt5 import QtCore 
from PyQt5 import QtWidgets
from Canvas import *
from Recommender import *
import resources

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent = None ):
        QtWidgets.QMainWindow.__init__(self, parent )
        self.setFocus()
        self.setWindowTitle("PowerPoint")

        self.cont = QtWidgets.QWidget(self)
        self.setCentralWidget(self.cont)
        self.canvas = Canvas(self)    
        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.statusBar().showMessage("Diapositive 1")
        self.statusBar().setStyleSheet("border: 1px solid; border-color:grey; background-color:white")

        # self.textEdit = QTextEdit(self.cont)
        # self.textEdit.setReadOnly(True)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(10,10,10,10)
        self.cont.setContentsMargins(10,10,10,10)

        # useless Canvas
        w = QtWidgets.QWidget() 
        w.setLayout( QtWidgets.QVBoxLayout())
        c = QLabel()
        self.preview = c
        c.setMinimumSize(150,100)
        c.setMaximumSize(150,100)
        c.setStyleSheet("border: 1px solid; border-color:grey; background-color:white")
        w.layout().addWidget(c)
        w.layout().addStretch()
        layout.addWidget(w)
        

        # separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Useful Canvas
        w = QMainWindow()
        w.setStyleSheet("border: 1px solid; border-color:grey; background-color:white")
        w.setCentralWidget(self.canvas)
        layout.addWidget(w)
        layout.setContentsMargins(11,0,11,0)

        bar = self.menuBar()
        css = """
        QMenuBar{
            Background: #b8442c;
            color:white;
            font-size:14px; }
        QMenuBar::item { color : white; }
        QMenuBar::item:selected { background: #dc5939; }
        QMenuBar::item:pressed { background: #dc5939;  }
        """
        bar.setStyleSheet(css)
        bar.resize(600,100)
        # File Menu
        fileMenu = bar.addMenu("File")
        fileMenu.addAction("&Picture", self.canvas.addImage)
        

        # Edit Menu
        editMenu = bar.addMenu("Edit")
        editMenu.addAction("&delete", self.canvas.deleteSelection, QKeySequence("Backspace"))
        editMenu.addSeparator()
        editMenu.addAction(QIcon(":/icons/copy.png"), "&Copy", self.canvas.copy_element, QKeySequence("Ctrl+C"))
        editMenu.addAction(QIcon(":/icons/paste.png"), "&Paste",  self.canvas.paste_element, QKeySequence("Ctrl+V"))
        editMenu.addAction(QIcon(":/icons/cut.png"), "&Cut", self.canvas.cut_element,  QKeySequence("Ctrl+X"))
        editMenu.addAction("&Duplicate",  self.canvas.duplicate_element, QKeySequence("Ctrl+D"))
        editMenu.addSeparator()
        editMenu.addAction("&Group",  self.canvas.group, QKeySequence("Ctrl+G"))
        editMenu.addAction("&UnGroup",  self.canvas.ungroup, QKeySequence("Ctrl+Shift+G"))
        editMenu.addSeparator()
        editMenu.addAction("&Randomize",  self.canvas.randomize, QKeySequence("Ctrl+R"))


        editMenu = bar.addMenu("Organiser")
        editMenu.addAction("&Align Top",  self.canvas.alignTop)
        editMenu.addAction("&Align Right",  self.canvas.alignRight)
        editMenu.addAction("&Align Left",  self.canvas.alignLeft)
        editMenu.addAction("&Align Bottom",  self.canvas.alignBottom)
        editMenu.addSeparator()
        editMenu.addAction(QIcon(":/icons/foreground.png"), "&Foreground",  self.canvas.deplaceLast)
        editMenu.addAction(QIcon(":/icons/background.png"), "&Background",  self.canvas.deplaceFirst)



        # Menu Color
        actPen = fileMenu.addAction(QIcon(":/icons/pen.png"), "&Pen color", self.pen_color, QKeySequence("Ctrl+P"))
        actBrush = fileMenu.addAction(QIcon(":/icons/brush.png"), "&Brush color", self.brush_color, QKeySequence("Ctrl+B"))

        colorToolBar = QToolBar("Color")
        self.addToolBar( colorToolBar )
        
        colorToolBar.addAction( actPen )
        colorToolBar.addAction( actBrush )

        # Menu des formes
        shapeMenu = bar.addMenu("Shape")
        actRectangle = shapeMenu.addAction(QIcon(":/icons/rectangle.png"), "&Rectangle", self.rectangle )
        actEllipse = shapeMenu.addAction(QIcon(":/icons/ellipse.png"), "&Ellipse", self.ellipse)
        actTriangle = shapeMenu.addAction(QIcon(":/icons/triangle.png"), "&Triangle", self.triangle)
        actFree = shapeMenu.addAction(QIcon(":/icons/free.png"), "&Free drawing", self.free_drawing)
        actSave = fileMenu.addAction(QIcon(":/image/images/save.png"), "&Save", self.save, QKeySequence("Ctrl+S"))
        # Toolbar des formes
        shapeToolBar = QToolBar("Shape")
        self.addToolBar( shapeToolBar )
        shapeToolBar.addAction( actRectangle )
        shapeToolBar.addAction( actEllipse )
        shapeToolBar.addAction( actTriangle )
        
        # Menu des modes
        modeMenu = bar.addMenu("Mode")
        actMove = modeMenu.addAction(QIcon(":/icons/move.png"), "&Move", self.move)
        actDraw = modeMenu.addAction(QIcon(":/icons/draw.png"), "&Draw", self.draw)
        actSelect = modeMenu.addAction(QIcon(":/icons/select.png"), "&Select", self.select)

        # Tool bar des modes
        modeToolBar = QToolBar("Navigation")
        self.addToolBar( modeToolBar )
        modeToolBar.addAction( actMove )
        modeToolBar.addAction( actDraw )
        modeToolBar.addAction( actSelect )

        # Menu des view
        viewMenu = bar.addMenu("View")
        actZoom_in = viewMenu.addAction(QIcon(":/icons/zoom-in.png"), "&Zoom-in", self.canvas.zoom_in)
        actZoom_out= viewMenu.addAction(QIcon(":/icons/zoom-out.png"), "&Zoom-out", self.canvas.zoom_out)
        # Tool bar des view
        viewToolBar = QToolBar("View")
        self.addToolBar( viewToolBar )
        viewToolBar.addAction( actZoom_in )
        viewToolBar.addAction( actZoom_out )
        self.cont.setLayout(layout)

        # Couleurs
        colorToolBar = QToolBar("Couleurs")
        self.colorToolButton = QToolButton()
        self.colorBorderToolButton = QToolButton()
        def setColor(color):
            pixmap = QPixmap(20,20)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            self.colorToolButton.setIcon(icon)
            self.canvas.set_color(color)
        
        def setColorBorder(color):
            pixmap = QPixmap(20,20)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            self.colorBorderToolButton.setIcon(icon)
            self.canvas.set_color_border(color)
        # Remplissage
        colorMenu = ColorMenu(self, setColor)

        self.colorToolButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.colorToolButton.setText("Fill")
        self.colorToolButton.setMenu(colorMenu)
        self.colorToolButton.setPopupMode(QToolButton.InstantPopup)
        colorToolBar.addWidget(self.colorToolButton)

        colorMenu = ColorMenu(self, setColorBorder)
        self.colorBorderToolButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.colorBorderToolButton.setText("Border")
        self.colorBorderToolButton.setMenu(colorMenu)
        self.colorBorderToolButton.setPopupMode(QToolButton.InstantPopup)
        colorToolBar.addWidget(self.colorBorderToolButton)
        self.addToolBar( colorToolBar )

        setColor(QColor(0, 255, 255))
        setColorBorder(QColor(0, 255, 255))

        # Permet d'ajouter les actions de déplacer l'objet selectionné
        self.addMoveFeature()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.canvas.move_element('Up')
        elif event.key() == Qt.Key_Down:
            self.canvas.move_element('Down')
        elif event.key() == Qt.Key_Left:
            self.canvas.move_element("Left")
        elif event.key() == Qt.Key_Right:
            self.canvas.move_element('Right')

    def addMoveFeature(self):
        # MoveElement
        scale = 3
        action =  QAction("MoveUp", self)
        action.setShortcuts(QKeySequence("Ctrl+Up"))
        action.triggered.connect(lambda: self.canvas.move_element("Up", scale))
        self.addAction(action)

        action =  QAction("MoveDown", self)
        action.setShortcuts(QKeySequence("Ctrl+Down"))
        action.triggered.connect(lambda: self.canvas.move_element("Down", scale))
        self.addAction(action)

        action =  QAction("MoveLeft", self)
        action.setShortcuts(QKeySequence("Ctrl+Left"))
        action.triggered.connect(lambda: self.canvas.move_element("Left", scale))
        self.addAction(action)

        action =  QAction("MoveRight", self)
        action.setShortcuts(QKeySequence("Ctrl+Right"))
        action.triggered.connect(lambda: self.canvas.move_element("Right", scale))
        self.addAction(action)


    ##############
    def pen_color(self):
        self.log_action("choose pen color")

    def brush_color(self):
        self.log_action("choose brush color")

    def rectangle(self):
        self.log_action("Shape mode: rectangle")
        self.canvas.setTool("rectangle")

    def ellipse(self):
        self.log_action("Shape Mode: circle")
        self.canvas.setTool("ellipse")

    def triangle(self):
        self.log_action("Shape Mode: circle")
        self.canvas.setTool("triangle")

    def free_drawing(self):
        self.log_action("Shape mode: free drawing")
        self.canvas.setTool("drawLines")

    def move(self):
        self.log_action("Mode: move")
        self.canvas.setMode('move')

    def draw(self):
        self.log_action("Mode: draw")
        self.canvas.setMode('draw')

    def select(self):
        self.log_action("Mode: select")
        self.canvas.setMode('select')

    def save(self):
        self.log_action("Saving canvas")
        image = self.canvas.getImage()
        image.save('image.jpg')

    def log_action(self, str):
        pass

# PowerPoint-Like Title Bar
class TitleBar(QDialog):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        css = """
        QWidget{
            Background: #b8442c;
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
            Background:#b8442c;
            font-size:11px;
        }
        QToolButton:hover{
            Background: #dc5939;
            font-size:11px;}
        """
        self.setAutoFillBackground(True)
        self.setBackgroundRole(QPalette.Highlight)
        self.setStyleSheet(css) 
        self.minimize=QToolButton(self)
        self.minimize.setIcon(QIcon("icons/remove.png"))
        self.maximize=QToolButton(self)
        self.maximize.setIcon(QIcon("icons/resize.png"))
        close=QToolButton(self)
        close.setStyleSheet("""QToolButton:hover{
            Background: #e81123;
            font-size:11px;
        }""")
        close.setIcon(QIcon("icons/close.png"))
        self.minimize.setMinimumSize(40,40)
        close.setMinimumSize(40,40)
        self.maximize.setMinimumSize(40,40)
        label=QLabel(self)
        label.setText("PowerPoint")
        self.setWindowTitle("Window Title")
        hbox=QHBoxLayout(self)
        hbox.addWidget(label)
        hbox.addWidget(self.minimize)
        hbox.addWidget(self.maximize)
        hbox.addWidget(close)
        hbox.insertStretch(1,500)
        hbox.setSpacing(0)
        hbox.setContentsMargins(10,0,0,0)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.maxNormal=False
        close.clicked.connect(self.close)
        self.minimize.clicked.connect(self.showSmall)
        self.maximize.clicked.connect(self.showMaxRestore)

    def showSmall(self):
        self.parent().showMinimized()

    def showMaxRestore(self):
        if(self.maxNormal):
            self.parent().showNormal()
            self.maxNormal= False
            self.maximize.setIcon(QIcon("icons/resize.png"))
        else:
            self.parent().showMaximized()
            self.maxNormal=  True
            self.maximize.setIcon(QIcon("icons/resize.png"))

    def close(self):
        QCoreApplication.quit()
        self.parent().close()

    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.parent().moving = True 
            self.parent().offset = event.pos()

    def mouseMoveEvent(self,event):
        if self.parent().moving: 
            self.parent().move(event.globalPos()-self.parent().offset)

# Frame with PowerPoint Title Bar
class PowerPoint(QFrame):
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
        self.window = MainWindow()
        vbox.addWidget(self.window)


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

class ColorAction(QtWidgets.QWidgetAction):
    colorSelected = pyqtSignal(QtGui.QColor)

    def __init__(self, parent):
        super(ColorAction, self).__init__(parent)
        widget = QtWidgets.QWidget(parent)
        layout = QtWidgets.QGridLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(2, 2, 2, 2)
        palette = self.palette()
        count = len(palette)
        rows = count // round(count ** .5)
        for row in range(rows):
            for column in range(count // rows):
                color = palette.pop()
                button = QtWidgets.QToolButton(widget)
                button.setAutoRaise(True)
                button.clicked.connect(
                    lambda checked, color=color: self.handleButton(color))
                pixmap = QtGui.QPixmap(16, 16)
                pixmap.fill(color)
                button.setIcon(QtGui.QIcon(pixmap))
                layout.addWidget(button, row, column)
        self.setDefaultWidget(widget)

    def handleButton(self, color):
        self.parent().hide()
        self.colorSelected.emit(color)

    def palette(self):
        palette = []
        for g in range(4):
            for r in range(4):
                for b in range(3):
                    palette.append(QtGui.QColor(
                        r * 255 // 3, g * 255 // 3, b * 255 // 2))
        return palette

class ColorMenu(QtWidgets.QMenu):
    def __init__(self, parent=None, setColor=None):
        super(ColorMenu, self).__init__("Colors", parent)
        self.colorAction = ColorAction(self)
        self.colorAction.colorSelected.connect(self.handleColorSelected)
        self.addAction(self.colorAction)
        self.addSeparator()
        self.addAction('Custom Color...', lambda: self.handleColorSelected(QColorDialog.getColor()))
        self.setColor = setColor

    def handleColorSelected(self, color):
        print(color)
        if self.setColor is not None:
            self.setColor(color)


if __name__=="__main__":
    app = QApplication(sys.argv)
    # Logiciel PowerPoint
    p = PowerPoint()
    l=QVBoxLayout(p.contentWidget())
    l.setContentsMargins(0,0,0,0)
    window = p.window
    p.show()

    # # Model
    # classif = LeNet2()
    # tak2Image2Image=True
    # process = classif.process_2image
    # if not tak2Image2Image:
    #     classif.fc1 = nn.Linear(1024,1024)
    #     process = classif.process_1image
    # classif.load_state_dict(torch.load("models/model_2image_all")) # loading parameters
    # model = Model(classif, process, tak2Image2Image=tak2Image2Image, classe_names = ["AlignBottom", "AlignLeft", 'AlignRight', 'AlignTop'])

    # classif = LeNet2()
    # tak2Image2Image=True
    # process = classif.process_2image
    # if not tak2Image2Image:
    #     classif.fc1 = nn.Linear(1024,1024)
    #     process = classif.process_1image
    # classif.load_state_dict(torch.load("models/model_2image_partial")) # loading parameters
    # model2 = Model(classif, process, tak2Image2Image=tak2Image2Image, classe_names = ["AlignBottom", "AlignLeft", 'AlignRight', 'AlignTop'])

    # model3 = HardCodedModel()
    model = HardCodedModel()
    # Recommender
    # LR = [Recommender(m, show_state=False, moving=False, title=t) for m,t in zip([model, model2, model3],["all", "partial", "hardcode"])]
    R = Recommender(model, show_state=False, moving=True, direction='right')
    window.canvas.logger.recommender = R
    
    app.exec_()