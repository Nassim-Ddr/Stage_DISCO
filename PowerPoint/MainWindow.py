import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from Canvas import *
from Recommender import *
import resources

class MainWindow(QMainWindow):
    def __init__(self, parent = None ):
        QMainWindow.__init__(self, parent )
        self.resize(600, 500)
        self.setFocus()
        self.setWindowTitle("PowerPoint")

        self.cont = QWidget(self)
        self.setCentralWidget(self.cont)
        self.canvas = Canvas(self)     

        self.textEdit = QTextEdit(self.cont)
        self.textEdit.setReadOnly(True)

        self.setStyleSheet("QMainWindow::titleBar { background-color: red; }")

        layout = QVBoxLayout()
        w = QMainWindow()
        w.setStyleSheet("border: 1px solid; border-color:grey; background-color:white")
        w.setCentralWidget(self.canvas)
        layout.addWidget(w)
        layout.addWidget(self.textEdit)

        bar = self.menuBar()
        #bar.setStyleSheet("background-color: #b8442c; color: white")
        bar.resize(600,100)
        # File Menu
        fileMenu = bar.addMenu("File")
        fileMenu.addAction("&Picture", self.canvas.addImage)
        

        # Edit Menu
        editMenu = bar.addMenu("Edit")
        editMenu.addAction("&delete", self.canvas.deleteSelection, QKeySequence("Backspace"))
        editMenu.addAction(QIcon(":/icons/copy.png"), "&Copy", self.canvas.copy_element, QKeySequence("Ctrl+C"))
        editMenu.addAction(QIcon(":/icons/paste.png"), "&Paste",  self.canvas.paste_element, QKeySequence("Ctrl+V"))
        editMenu.addAction(QIcon(":/icons/cut.png"), "&Cut", self.canvas.cut_element,  QKeySequence("Ctrl+X"))
        editMenu.addAction("&Duplicate",  self.canvas.duplicate_element, QKeySequence("Ctrl+D"))
        editMenu.addAction("&Group",  self.canvas.group, QKeySequence("Ctrl+G"))
        editMenu.addAction("&UnGroup",  self.canvas.ungroup, QKeySequence("Ctrl+Shift+G"))
        editMenu.addAction("&Randomize",  self.canvas.randomize, QKeySequence("Ctrl+R"))


        editMenu = bar.addMenu("Organiser")
        editMenu.addAction("&Align Top",  self.canvas.alignTop)
        editMenu.addAction("&Align Right",  self.canvas.alignRight)
        editMenu.addAction("&Align Left",  self.canvas.alignLeft)
        editMenu.addAction("&Align Bottom",  self.canvas.alignBottom)


        # Menu Color
        colorMenu = bar.addMenu("Color")
        actPen = fileMenu.addAction(QIcon(":/icons/pen.png"), "&Pen color", self.pen_color, QKeySequence("Ctrl+P"))
        actBrush = fileMenu.addAction(QIcon(":/icons/brush.png"), "&Brush color", self.brush_color, QKeySequence("Ctrl+B"))
        
        actRed = colorMenu.addAction("Rouge")
        actRed.triggered.connect(lambda: self.canvas.set_color(QColor(Qt.red)))
        colorMenu.addAction(actRed)
        actBlue = colorMenu.addAction("Bleu")
        actBlue.triggered.connect(lambda: self.canvas.set_color(QColor(Qt.blue)))
        colorMenu.addAction(actBlue)
        actGreen = colorMenu.addAction("Vert")
        actGreen.triggered.connect(lambda: self.canvas.set_color(QColor(Qt.green)))
        colorMenu.addAction(actGreen)
        actOther = colorMenu.addAction("Autre")
        actOther.triggered.connect(lambda: self.canvas.set_color(QColorDialog.getColor()))
        colorMenu.addAction(actOther)

        colorToolBar = QToolBar("Color")
        self.addToolBar( colorToolBar )
        
        colorToolBar.addAction( actPen )
        colorToolBar.addAction( actBrush )

        # Menu des formes
        shapeMenu = bar.addMenu("Shape")
        actRectangle = shapeMenu.addAction(QIcon(":/icons/rectangle.png"), "&Rectangle", self.rectangle )
        actEllipse = shapeMenu.addAction(QIcon(":/icons/ellipse.png"), "&Ellipse", self.ellipse)
        actFree = shapeMenu.addAction(QIcon(":/icons/free.png"), "&Free drawing", self.free_drawing)
        actSave = fileMenu.addAction(QIcon(":/image/images/save.png"), "&Save", self.save, QKeySequence("Ctrl+S"))
        # Toolbar des formes
        shapeToolBar = QToolBar("Shape")
        self.addToolBar( shapeToolBar )
        shapeToolBar.addAction( actRectangle )
        shapeToolBar.addAction( actEllipse )

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

    def lasso_select(self):
        self.log_action("Mode: lasso")
        self.canvas.setMode('lasso')

    def scriboli(self):
        self.log_action("Mode: scriboli")
        self.canvas.setMode('scriboli')

    def save(self):
        self.log_action("Saving canvas")
        image = self.canvas.getImage()
        image.save('image.jpg')

    def log_action(self, str):
        content = self.textEdit.toPlainText()
        self.textEdit.setPlainText( content + "\n" + str)
    

if __name__=="__main__":
    app = QApplication(sys.argv)
    # Logiciel PowerPoint
    window = MainWindow()
    window.show()
    # Recommender
    R = Recommender("models/model")
    window.canvas.logger.recommender = R
    R.show()

    app.exec_()
    window.canvas.logger.file.close()
