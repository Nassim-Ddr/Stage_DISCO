from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, transforms
import torchvision.transforms as T
import matplotlib.pyplot as plt

from PIL import Image

class Recommender(QMainWindow):
    def __init__(self, model, max_size_memory = 5):
        super().__init__()
        # Interface du recommender
        self.setWindowTitle("Assistant qui bourre le pantalon")
        b = False
        if b:
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TranslucentBackground, True)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addStretch()
        self.text = QLabel(self)
        self.text.setText("HELLO WORLD")
        self.text.setStyleSheet("margin-left: 10px; border-radius: 20px; background: white; color: #4A0C46;")
        self.text.setAlignment(Qt.AlignCenter)
        self.text.setMinimumSize(QSize(200,100))

        layout.addWidget(self.text)

        # variable du recommender
        self.model = Model(model, ["AlignBottom", "AlignLeft", 'AlignRight', 'AlignTop'])
        self.memory = []
        self.max_size_memory = max_size_memory

        self.initUI()

    def initUI(self):
        self.resize(640, 480)
        QTimer.singleShot(1, self.topLeft)
        self.show()

    def topLeft(self):
        # no need to move the point of the geometry rect if you're going to use
        # the reference top left only
        topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint)

    def paintEvent(self, event=None):
        painter = QPainter(self)
        painter.setOpacity(0.0)
        painter.setBrush(Qt.white)
        painter.setPen(QPen(Qt.white))   
        painter.drawRect(self.rect())

    def update(self, state):
        print("Get Image")
        state = self.QImageToCvMat(state)
        print('TO Numpy')
        m_size = len(self.memory)
        for i in range(m_size): 
            s = self.memory[i]
            # cree la donnee a predire
            print('Predicting')
            pred_command, confiance = self.model.predict(s, state) 
            print("Prediction: ", i)
            self.setText(f'Predicted Command: {pred_command}\nConfiance: {confiance}')
            break
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(state)
        if m_size >= self.max_size_memory: self.memory.pop(0)

    def setText(self, text):
        self.text.setText(text)

    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''
        incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr
    
class Model():
    def __init__(self, model, classe_names = None):
        self.model = LeNet()
        self.model.load_state_dict(torch.load(model))
        self.classe_names = classe_names

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            crop(),
            transforms.Resize((64, 64)),
        ])
    
    # a, b (np.array)
    # return (prediction, confiance)
    def predict(self, a, b):
        s = np.array(b).astype(np.uint8)[1:,:,:3]
        s = Image.fromarray(s)
        x = self.image_transform(s).reshape((1,3,64,64))
        #plt.imshow(T.ToPILImage()(x[0]))
        #plt.show()
        output = self.model(x)
        index = output.argmax(1)
        if self.classe_names is not None:
            index = self.classe_names[index]
        return str(index), output.softmax(dim=1).max()

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
        print(img.mean(0))
        x1,x2,y1,y2 = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return img[:, x1:x2,y1:y2]
    
    def __repr__(self):
        return self.__class__.__name__+'()'



