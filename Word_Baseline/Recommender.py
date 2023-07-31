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
from sklearn.preprocessing import normalize

class Recommender(QWidget):
    def __init__(self, model,texteditor, max_size_memory = 10, hardCoded = False):
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

        # memory contenant les donnees pour le modele appris
        self.memory = []
        # memory contenant les données pour le modele harcode
        self.memory2 = []

        # la taille max des memoires
        self.max_size_memory = max_size_memory
        self.modelHard = HardCodedModel(texteditor,historySize=max_size_memory)
        # variable du recommender
        # self.model = Model(model, ["MoveWR","MoveWL","MoveHome","MoveEnd","Tab","WordDel","Replace","SelectWR","SelectWL","SelectAll"])
        self.model = Model(model, ["WriteWord","CopyPaste","WordDel","Search&Replace"])
            

    def update(self, state,stateHardcode,texteditor):
        #state, label = state
        m_size = len(self.memory)
        ok = True
        for i in range(m_size): 
            s = self.memory[i]
            if np.sum(s-state) == 0 :
                continue
            # cree la donnee a predire
            pred_command, confiance = self.model.predict(s, state) 
            if (pred_command != "WriteWord"):
                self.setText(f'Predicted Command: {pred_command}\nConfiance: {confiance}')
            elif(pred_command == "WriteWord"):
                if i != m_size-1:
                    continue
                else:
                    break
            ok = False
            break
        if ok :
            self.updateHardCoded(stateHardcode)
        else :
            self.memory2.append(stateHardcode)
            m_size2 = len(self.memory2)
            if m_size2 >= self.max_size_memory: self.memory2.pop(0)
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory.append(state)
        if m_size >= self.max_size_memory: self.memory.pop(0)
    
    
    
    def updateHardCoded(self, state,texteditor):
        # state, label = state
        m_size = len(self.memory2)
        for i in range(m_size): 
            s = self.memory2[i]
            if np.sum(s-state) == 0:
                continue
            if i == m_size-1:
                break
            # cree la donnee a predire
            pred_command = self.modelHard.predict(s, state,texteditor) 
            self.setText(pred_command)
            break
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory2.append(state)
        if m_size >= self.max_size_memory: self.memory2.pop(0)

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
        x = a-b
        x = x.reshape((1,len(x)))
        anotherX = normalize(x)
        #anotherX = np.array([anotherX]).astype("float32")
        anotherX = anotherX.astype("float32")
        anotherX = torch.from_numpy(anotherX)
        output = self.model(anotherX)
        index = output.argmax(1)
        if self.classe_names is not None:
            index = self.classe_names[index]
        print("output: ", output)
        print("index: ", index[0])
        return str(index), output.softmax(dim=1).max()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2881, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits