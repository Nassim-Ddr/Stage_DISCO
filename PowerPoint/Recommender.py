from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, transforms
import torchvision.transforms as T
from PIL import Image


from PyQt5.QtPrintSupport import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
        self.text.setStyleSheet("margin-left: 10px; border-radius: 20px; background: white; color: #4A0C46;")
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setMinimumSize(QSize(200,100))
        layout.addWidget(self.text)

        # Affiche état
        self.C = FigureCanvas(plt.figure( figsize=(5, 3)))
        layout.addWidget(self.C)
        self.ax1, self.ax2, self.ax3 = self.C.figure.subplots(1,3)
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # variable du recommender
        self.model = Model(model, ["AlignBottom", "AlignLeft", 'AlignRight', 'AlignTop'])
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

    def update(self, state):
        state = self.QImageToCvMat(state)
        m_size = len(self.memory)
        if m_size > 0:
            preds_conf = np.array([self.model.predict(s, state) for s in self.memory])
            index = np.argmax(preds_conf[:,1])
            pred_command, confiance = preds_conf[index]
            self.setText(f'Predicted Command: {pred_command}\nConfiance: {confiance}')
            self.showState(self.memory[index], state)
            
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(state)
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

    
    
class Model():
    def __init__(self, model, classe_names = None):
        self.model = LeNet()
        self.model.load_state_dict(torch.load(model))
        self.classe_names = classe_names
        self.process = Preprocessing()

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            crop(),
            transforms.Resize((64, 64)),
        ])
    
    # a, b (np.array)
    # return (prediction, confiance)
    def predict(self, a, b):
        b = self.process.getOnlyMovingObject(a,b)
        s = Image.fromarray(b)
        x = self.image_transform(s).reshape((1,3,64,64))
        output = self.model(x)
        index = output.argmax(1)
        if self.classe_names is not None:
            index = self.classe_names[index]
        return index, output.softmax(dim=1).max().item()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class crop(object):
    def __call__(self, img):
        a = np.where(img.mean(0) != 1)
        try:
            x1,x2,y1,y2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            h,w = img[:, x1:x2,y1:y2].shape[1:]
            if h<64 or w<64: return img
            return img[:, x1:x2,y1:y2]
        except:
            return img
    
    def __repr__(self):
        return self.__class__.__name__+'()'
    

class Preprocessing():
    def getOnlyMovingObject(self, a, b):
        r = b.copy()
        x,y = np.where(np.mean(a-b, axis=2)<=0)
        r[x,y,:] = 255
        return r



