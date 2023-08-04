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
from Model import *

class Recommender(QMainWindow):
    def __init__(self, model, max_size_memory = 5, parent = None):
        QMainWindow.__init__(self, parent )
        # Interface du recommender
        self.setWindowTitle("Assistant qui bourre le pantalon")
        b = False
        if b:
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(QSize(300,300))
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        # Affichage de la recommandation
        self.text = QLabel(self.container)
        self.text.setText("HELLO WORLD")
        self.text.setStyleSheet("margin-left: 10px; border-radius: 20px; background: white; color: #4A0C46; font-size:15px")
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setMinimumSize(QSize(200,100))
        layout.addWidget(self.text)

        # Affiche état
        self.C = FigureCanvas()
        layout.addWidget(self.C)
        self.ax1, self.ax2, self.ax3 = self.C.figure.subplots(1,3)
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # variable du recommender
        #self.model = Model(model, ["AlignBottom", "AlignLeft", 'AlignRight', 'AlignTop'])
        self.model = HardCodedModel()
        self.memory = []
        self.max_size_memory = max_size_memory

        self.initUI()
        self.setCentralWidget( self.container )


    def initUI(self):
        QTimer.singleShot(1, self.topLeft)
        self.show()

    def topLeft(self):
        # no need to move the point of the geometry rect if you're going to use
        # the reference top left only
        topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint + QPoint(50,50))
        pass

    def paintEvent(self, _):
        # painter = QPainter(self)
        # painter.setOpacity(0.0)
        # painter.setBrush(Qt.white)
        # painter.setPen(QPen(Qt.white))   
        # painter.drawRect(self.rect())
        pass

    def update(self, state, autre=None):
        state = self.QImageToCvMat(state)
        m_size = len(self.memory)
        if m_size > 0:
            #preds_conf = np.array([self.model.predict(s, state) for s in self.memory])
            #index = np.argmax(preds_conf[:,1])
            #pred_command, confiance = preds_conf[index]
            pred_command, confiance = self.model.predictForeorBackground(self.memory[-1], autre), "Tellement confiant"
            self.setText(f'Predicted Command: {pred_command}\nConfiance: {confiance}')
            #self.showState(self.memory[index], state)
            
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(autre)
        if m_size >= self.max_size_memory: self.memory.pop(0)

    def setText(self, text):
        self.text.setText(text)

    def QImageToCvMat(self, image):
        image.save(f'./images/state.jpg')
        image = Image.open("./images/state.jpg")
        image = np.asarray(image)
        return image
    
    def showState(self, a, b):
        # display transition between b
        for ax, title in zip([self.ax1,  self.ax2,  self.ax3],  ["Before", "After", 'Input to Model']):
            ax.clear()
            ax.tick_params(left = False, right = False , labelleft = False ,
                           labelbottom = False, bottom = False)
            ax.title.set_text(title)

        self.ax1.imshow(a)
        self.ax2.imshow(b)
        self.ax3.imshow(self.model.process.getOnlyMovingObject(a,b))
        self.C.draw()


