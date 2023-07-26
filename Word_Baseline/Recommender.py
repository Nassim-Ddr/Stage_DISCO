from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, transforms
from PIL import Image
from HardCodedModel import HardCodedModel

class Recommender(QWidget):
    def __init__(self, model, max_size_memory = 2, hardCoded = False):
        super().__init__()
        # Interface du recommender
        self.setWindowTitle("Assistant qui bourre le pantalon")
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addStretch()
        
        #self.setStyleSheet('background-color: lightblue;')
        self.setLayout(layout)
        self.text = QTextEdit(self)
        layout.addWidget(self.text)

        if hardCoded :
            self.model = HardCodedModel()

        else :
            # variable du recommender
            self.model = Model(model, ["MoveWR","MoveWL","MoveHome","MoveEnd","Tab","WordDel","Replace","SelectWR","SelectWL","SelectAll"])
            self.memory = []
            self.max_size_memory = max_size_memory

    def update(self, state):
        state, label = state
        m_size = len(self.memory)
        for i in range(m_size): 
            s = self.memory[i]
            # cree la donnee a predire
            pred_command, confiance = self.model.predict(s, state) 
            self.setText(f'Predicted Command: {pred_command}\nConfiance: {confiance}')
            break
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(state)
        if m_size >= self.max_size_memory: self.memory.pop(0)
    
    def updateHardCoded(self, state,texteditor):
        state, label = state
        m_size = len(self.memory)
        for i in range(m_size): 
            s = self.memory[i]
            # cree la donnee a predire
            pred_command = self.model.predict(s, state,texteditor) 
            self.setText(pred_command)
            break
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(state)
        if m_size >= self.max_size_memory: self.memory.pop(0)

    def setText(self, text):
        self.text.setPlainText(text)

class Model():
    def __init__(self, model, classe_names = None):
        self.model = NeuralNet()
        self.model.load_state_dict(torch.load(model))
        self.model.eval()
        self.classe_names = classe_names

    # a, b (np.array)
    # return (prediction, confiance)
    def predict(self, a,b):
        #x = np.hstack((a,b))
        x = b-a
        x = np.array([x]).astype("float32")
        x = torch.from_numpy(x)
        print(x.shape)
        output = self.model(x)
        index = output.argmax(1)
        if self.classe_names is not None:
            index = self.classe_names[index]
        print("output: ", output)
        print("index: ", index[0])
        return str(index), output.max()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanH_stack = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_tanH_stack(x)
        return logits