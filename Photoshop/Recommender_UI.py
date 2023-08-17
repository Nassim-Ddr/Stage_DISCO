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
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(QSize(250,150))
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        # Affichage de la recommandation
        self.text = QLabel(self.container)
        self.text.setText("HELLO WORLD")
        self.text.setStyleSheet("margin-left: 10px; border-radius: 20px; background: white; color: #4A0C46; font-size:12px")
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setMinimumSize(QSize(200,100))
        layout.addWidget(self.text)


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
        topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint + QPoint(- self.size().width(),50))
        self.timer.start()
        pass

    def update(self, text, autre=None):
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
        self.move(self.pos() + QPoint(5,0))

    # Normal move
    def initMove(self, waitTime = 2000):
        self.timer.setInterval(10)
        self.move(self.pos() + QPoint(self.mode*5,0))
        maxRight = QApplication.desktop().availableGeometry().left()
        if self.mode == 1:
            if self.pos().x() >= maxRight+10:
                self.mode = -1
                self.timer.setInterval(waitTime)
        else:
            if self.pos().x() <= maxRight - self.size().width():
                self.mode = 1
                self.timer.stop()