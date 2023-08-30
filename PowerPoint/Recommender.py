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

# Recommender: interface qui affiche les prédictions du modèle
# model: qui fait les prédictions sur les états de l'application
# max_size_memory: taille de l'historique des états
# parent: parent du widget (optionnel)
# show_state: affiche ou pas les états du résultat final
# moving: déplace ou non le widget pour cacher et sort si on a fait une prédiction
class Recommender(QMainWindow):
    def __init__(self, model, max_size_memory = 5, parent = None, show_state = False, moving= True, direction="right", title = "Assistant PowerPoint"):
        QMainWindow.__init__(self, parent )
        # Interface du recommender
        b = True
        if b:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMinimumSize(QSize(280*2,250))
        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setSpacing(0)
        layout.setContentsMargins(2,2,2,2)
        # Title bar
        label = QLabel(title, self.container)
        label.setStyleSheet(f"""
            Background: #b8442c;
            color:white;font:24px bold;
            font-weight:bold;
            height: 11px;""")
        label.setFixedHeight(60)
        label.setIndent(10)
        layout.addWidget(label)
        
        # Affichage de la recommandation
        self.text = QLabel("HELLO WORLD", self.container)
        self.text.setStyleSheet("""
                                border: black;
                                background: #efefef; 
                                color: #4A0C46; 
                                font-size:24px;
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

        # variable du recommender
        self.model = model
        self.memory = []
        self.count = 1
        self.max_size_memory = max_size_memory
        self.prev_recommendation = "Rien du Tout"
    
        self.initUI()
        self.setCentralWidget( self.container )

        # Timer
        self.direction = direction
        self.moving = moving
        if self.moving:
            self.timer = QTimer(self)
            self.timer.setInterval(10)
            self.timer.timeout.connect(self.initMove)
            self.mode = 1


    def initUI(self):
        QTimer.singleShot(1, self.topLeft)
        self.show()

    diff = 0
    def topLeft(self):
        # no need to move the point of the geometry rect if you're going to use
        # the reference top left only
        G = QApplication.desktop().availableGeometry()
        topLeftPoint = G.bottomLeft()  if self.direction == 'left' else G.bottomRight() -QPoint(self.size().width(), 0)
        self.move(topLeftPoint + QPoint(0,-self.size().height() - 10 - Recommender.diff))
        if self.moving:
            self.timer.start()
        Recommender.diff += self.size().height() - 10


    def update(self, state, autre=None, command = None):
        print(" =========== Predicting ? =============")
        m_size = len(self.memory)
        if m_size > 0:
            if isinstance(self.model, HardCodedModel):
                pred_command, confiance = self.model.predict(self.memory[-1], autre), "Tellement confiant"
                if pred_command != 'Rien du Tout': 
                    if command == pred_command or self.prev_recommendation == pred_command: 
                        print(f"Filtered command: {pred_command}")
                    else:
                        self.setText(pred_command)
                        self.prev_recommendation = pred_command
                        if self.moving:
                            self.mode = 1
                            self.timer.start()
                else:
                    self.prev_recommendation = "Rien du Tout"
            else:
                state = self.QImageToCvMat(state)
                preds_conf = np.array([self.model.predict(s, state) for s in self.memory])
                index = np.argmax(preds_conf[:,1])
                index = -1
                pred_command, confiance = preds_conf[index]
                self.setText(pred_command, confiance)

                if self.showingState: self.showState(self.memory[index], state)
                if self.moving:
                    self.mode = 1
                    self.timer.start()

            
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        if isinstance(self.model, HardCodedModel): self.memory.append(autre)
        else: self.memory.append(state)
        if m_size >= self.max_size_memory: self.memory.pop(0)

    def setText(self, cmd, confiance = None):
        self.count += 1
        r =f'<div style="font-weight:600; color:#aa0000;">{cmd}</div>'
        c =f'<div style="font-weight:600; color:#aa0000;">{confiance}</div>'
        if confiance is None:
            self.text.setText(f'Recommendation n°{self.count}:\n{r}')
        else:
            self.text.setText(f'Recommendation n°{self.count}: {r}\nConfiance: {c}')

    def QImageToCvMat(self, image):
        if not image.save("images/state.png"):
            print("Warning: saving failed image state.png ")
        image = Image.open("./images/state.jpg")
        image = np.asarray(image)
        return image
    
    def showState(self, a, b):
        # display transition between b
        z = zip([self.ax1,  self.ax2,  self.ax3],  ["Before", "After", 'Input to Model'])
        # z = zip([self.ax1,  self.ax2],  ["Before", "After"])
        for ax, title in z:
            ax.clear()
            ax.tick_params(left = False, right = False , labelleft = False ,
                           labelbottom = False, bottom = False)
            ax.title.set_text(title)

        self.ax1.imshow(a)
        self.ax2.imshow(b)
        self.ax3.imshow(T.ToPILImage()(self.model.input(a,b)[0]))
        self.C.draw()

    # Train move
    def translate(self):
        nb_step = 100
        maxRight = QApplication.desktop().availableGeometry().right()
        right = self.pos().x()
        if right >= maxRight:
            self.topLeft()
        self.move(self.pos() + QPoint(5,0))

    # Normal move
    def initMove(self, waitTime = 4000):
        self.timer.setInterval(10)
        direction = 1 if self.direction=='left' else -1
        self.move(self.pos() + QPoint(self.mode*5*direction,0))
        if self.direction == 'left':
            maxRight = QApplication.desktop().availableGeometry().left()
            if self.mode == 1:
                if self.pos().x() >= maxRight+10:
                    self.mode = -1
                    self.timer.setInterval(waitTime)
            else:
                if self.pos().x() <= maxRight - self.size().width():
                    self.mode = 1
                    self.timer.stop()
        else:
            maxRight = QApplication.desktop().availableGeometry().right() - self.size().width()
            if self.mode == 1:
                if self.pos().x() <= maxRight-10:
                    self.mode = -1
                    self.timer.setInterval(waitTime)
            else:
                if self.pos().x() >= maxRight + self.size().width():
                    self.mode = 1
                    self.timer.stop()



