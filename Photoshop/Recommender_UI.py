from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,  QLabel, QVBoxLayout
from PyQt5.QtGui import QKeySequence, QColor, QPalette, QIcon, QPixmap
from PyQt5.QtCore import Qt, QSize, QTimer, QPoint


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision.transforms as T
from PIL import Image

from PyQt5.QtPrintSupport import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Recommender: interface qui affiche les prédictions du modèle
class Recommender_UI(QMainWindow):
    def __init__(self, parent = None, show_state = False):
        QMainWindow.__init__(self, parent )
        # Interface du recommender
        self.setWindowTitle("Assistant")
        b = True
        if b:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(QSize(400,200))
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setSpacing(0)
        # Title bar
        label = QLabel("Assistant Photoshop", self.container)
        label.setStyleSheet(f"""
            Background: #6a6a6a;
            color:white;font:20px bold;
            font-weight:bold;
            height: 11px;""")
        label.setFixedHeight(40)
        label.setIndent(10)
        layout.addWidget(label)
        # Affichage de la recommandation
        self.text = QLabel("HELLO WORLD", self.container)
        self.text.setStyleSheet("""
                                background: #efefef; 
                                color: black; 
                                font-size:20px;
                                font-weight: 500;""")
        self.text.setIndent(20)
        self.text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        layout.addWidget(self.text)

        self.showingState = show_state
        if show_state:
            self.setMinimumSize(QSize(500,350))
            # Affiche état
            self.C = FigureCanvas()
            self.C.minumumSizeHint()
            layout.addWidget(self.C)
            self.ax1, self.ax2, self.ax3 = self.C.figure.subplots(1,3)
            # self.ax1, self.ax2 = self.C.figure.subplots(1,2)
            self.ax1.axis('off')
            self.ax2.axis('off')
            self.ax3.axis('off')


        self.initUI()
        self.setCentralWidget( self.container )

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.initMove)
        self.mode = 1


    def initUI(self):
        QTimer.singleShot(1, self.topLeft)
        self.show()

    def topLeft(self):
        # no need to move the point of the geometry rect if you're going to use
        # the reference top left only
        G = QApplication.desktop().availableGeometry()
        topLeftPoint = G.topRight() +QPoint(0, self.size().width() -50)
        self.move(topLeftPoint + QPoint(0,-self.size().height() - 10))
        self.timer.start()

    def update(self, text, autre=None):
        print("started")
        self.setText(text)
        self.mode = 1
        self.timer.start()

    def setText(self, text):
        self.text.setText(text)

    # Train move
    def translate(self):
        nb_step = 100
        maxRight = QApplication.desktop().availableGeometry().right()
        right = self.pos().x()
        if right >= maxRight:
            self.topLeft()
        self.move(self.pos() + QPoint(10,0))

    # Normal move
    def initMove(self, waitTime = 4000):
        self.timer.setInterval(10)
        self.move(self.pos() + QPoint(self.mode*5*(-1),0))
        maxRight = QApplication.desktop().availableGeometry().right() - self.size().width()
        if self.mode == 1:
            if self.pos().x() <= maxRight-10:
                self.mode = -1
                self.timer.setInterval(waitTime)
        else:
            if self.pos().x() >= maxRight + self.size().width():
                self.mode = 1
                self.timer.stop()