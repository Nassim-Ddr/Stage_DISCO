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

class Recommender(QMainWindow):
    def __init__(self, model,texteditor, max_size_memory = 10, hardCoded = False):
        super().__init__()
        # Interface du recommender
        self.setWindowTitle("Assistant qui bourre le pantalon")
        
        b = True
        if b:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
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

        # memory contenant les donnees pour le modele appris
        self.memory = []
        # memory contenant les données pour le modele harcode
        self.memory2 = []
        self.texmem=[]

        # la taille max des memoires
        self.max_size_memory = max_size_memory
        self.modelHard = HardCodedModel(texteditor,historySize=max_size_memory)
        # variable du recommender
        # self.model = Model(model, ["MoveWR","MoveWL","MoveHome","MoveEnd","Tab","WordDel","Replace","SelectWR","SelectWL","SelectAll"])
        self.commands = ["WriteWord","CopyPaste (CTRL + C -> CTRL + V)","WordDel (CTRL + Backspace)","Search&Replace (CTRL + R)"]
        self.model = Model(model, self.commands)
        self.hardCodedCommands = ["CTRL+ A (SelectAll)", 
        "Shift + Fin (End) Button", 
        "CTRL + Shift + Right", 
        "Fin (End)", 
        "CTRL + Right", 
        "Shift + Home", 
        "CTRL + Shift + Left",
        "Home",
        "CTRL + Left"
        ]
        self.recommendThreshold = np.zeros(len(self.hardCodedCommands))
        self.recommendThresholdML = np.zeros(4)
        self.stopRecom = 2

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
        self.move(topLeftPoint + QPoint(- self.size().width(),100))
        self.timer.start()
        pass
        

    def update(self, state,stateHardcode,texteditor,command):
        
        #state, label = state
        m_size = len(self.memory)
        ok = True
        
        # on compare l'etat courant avec les etats passes
        for i in range(m_size): 
            s = self.memory[i]
            sumstate = np.sum(s-state)
            # si l'etat correspondant au vecteur bow ne change pas on ignore
            if sumstate == 0 :
                continue
            # cree la donnee a predire
            pred_command, confiance = self.model.predict(s, state) 
            
            # On veut eviter de continuer a recommander
            ind = self.commands.index(pred_command)
            #print(f' La commande utilisée est : {command} et la prédiction est : {pred_command}')
            if pred_command == command:
                self.recommendThresholdML[ind] = self.recommendThresholdML[ind] + 1
            
            # la confiance doit etre > 95% (meme si ce n'est pas forcement un bon facteur)
            if (pred_command != "WriteWord" and confiance > 0.95 and command != pred_command):
                # Pour eviter de recommander des la premiere suppression de mot
                if pred_command == "WordDel (CTRL + Backspace)" and sumstate < 4 :
                    break
                
                # nous avons suffisament recommande on n'affiche rien (cela sera pareil pour le modele hard code)
                if self.recommendThresholdML[ind] >= self.stopRecom:
                    self.setText("")
                    ok = False
                    break
                
                # on affiche la commande predite avec la confiance
                self.setText(f'Predicted Command: {pred_command}\n La confiance du besoin de cette recommandation est : {confiance}')

                self.memory.clear()
                self.memory2.clear()
                self.texmem.clear()

                self.texmem.append(texteditor.toPlainText())
                self.memory.append(state)
                self.memory2.append(stateHardcode)

                self.timer.start()

                return
            # on filtre le cas ou l'utilisateur ne fait que ecrire
            elif(pred_command == "WriteWord"):
                if i != m_size-1:
                    continue
                else:
                    break
            ok = False
            break
        # dans le cas ou aucune commande n'est trouvee avec le vecteur bow on essaye avec la selection
        if ok :
            self.updateHardCoded(stateHardcode,command,texteditor)
        # sinon on memorise juste l'etat en retirant si la memoire est trop grande
        else :
            self.memory2.append(stateHardcode)
            m_size2 = len(self.memory2)
            if m_size2 >= self.max_size_memory: self.memory2.pop(0)
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.texmem.append(texteditor.toPlainText())
        self.memory.append(state)
        if m_size >= self.max_size_memory: 
            self.memory.pop(0)
            self.texmem.pop(0)
    
    
    
    def updateHardCoded(self, state,command,texteditor):
        # state, label = state
        m_size = len(self.memory2)
        keepinmind = texteditor.toPlainText()
        for i in range(m_size): 
            if self.texmem[i] != keepinmind:
                continue

            s = self.memory2[i]

            if np.sum(s-state) == 0:
                self.memory2 = self.memory2[:i]
                return
            if i == m_size-1:
                break
            # cree la donnee a predire

            
            pred_command = self.modelHard.predict(s, state)
            if (pred_command is None):
                continue
            ind = self.hardCodedCommands.index(pred_command)

            # Si la commande trouvee est deja utilisee suffisament on evite de recommander 
            if self.recommendThreshold[ind] >= self.stopRecom:
                self.setText("")
                break

            # On veut eviter de continuer a recommander trop
            if pred_command == command:
                ind = self.hardCodedCommands.index(pred_command)
                self.recommendThreshold[ind] = self.recommendThreshold[ind] + 1
            
            self.setText(pred_command)
            self.memory2.clear()
            self.timer.start()
            break
        # ajoute l'etat precedent
        # supprime si la liste est trop grande
        self.memory2.append(state)
        if m_size >= self.max_size_memory: self.memory2.pop(0)

    def setText(self, text):
        self.text.setText(text)

    
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
        # On doit formatter et normaliser (bonjour ScikitLearn)
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

# Le modele suivant est experimental, il existe potentiellement un meilleur modele
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