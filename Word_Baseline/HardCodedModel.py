import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import sklearn
import warnings
from sklearn.neighbors import KNeighborsClassifier as KNN
from time import time
import pandas as pd


# Some sort of decision tree manually done (only for selection so not an issue)
class HardCodedModel():
    def __init__(self,texteditor,historySize):
        self.limit = historySize
        self.texteditor = texteditor
        # self.hardCodedCommands = ["SelectAll (CTRL+ A)", 
        # "SelectToEndOfLine (Shift + Fin/End)", 
        # "SelectRightWord (CTRL + Shift + Right)", 
        # "MoveToEndOfLine (Fin/End)", 
        # "JumpRightWord (CTRL + Right)", 
        # "SelectToStartOfLine (Shift + Home)", 
        # "SelectLeftWord (CTRL + Shift + Left)",
        # "MoveToStartOfLine (Home)",
        # "JumpLeftWord (CTRL + Left)"
        # ]
        
        self.hardCodedCommands = ["SelectAll", 
        "SelectToEndOfLine", 
        "SelectRightWord", 
        "MoveToEndOfLine", 
        "JumpRightWord", 
        "SelectToStartOfLine", 
        "SelectLeftWord",
        "MoveToStartOfLine",
        "JumpLeftWord"
        ]
        
    def predict(self, oldState, newState):
        #inputData, label = inputData[:len(inputData)-1], inputData[-1]
        #past, present = inputData[:5], inputData[5:]
        # 2 Vecteurs d'etats
        past, present = oldState,newState
        # oldPos : la position passee, 
        # oldline1 : ligne du curseur precedemment, 
        # oldcolumn1 : pareil mais pour la colonne, 
        # oldselStart1 : position de départ de la selection passee
        # oldselEnd1 : position de fin de la selection passee
        # toutes les valeurs sont numeriques
        oldPos, oldline1, oldcolumn1, oldselStart1, oldselEnd1= past
        # positions pour l'etat courant
        newPos, newLine1, newColumn1, newselStart1, newselEnd1 = present

        #print(f'Old selection = {oldselStart1} {oldselEnd1} and new Selection = {newselStart1} {newselEnd1}')

        cursor = self.texteditor.textCursor()
        
        # Si tout le texte est selectionne, on peut suggerer la commande de selectionner tout le document
        end = len(self.texteditor.toPlainText())
        if (newselStart1 == 0 and newselEnd1 == end):
                # Si l'utilisateur a utilise la commande un certain nombre de fois on n'a plus besoin de demander
                return self.hardCodedCommands[0]

        # On va plus loin dans le document
        isEnd = self.check_cursorEnd(cursor)
        isStart = self.check_cursorStart(cursor)
        if oldPos < newPos :  
            if newPos-oldPos > 3 :
                if oldselStart1 == newselStart1 and oldselEnd1 < newselEnd1:
                    if isEnd :
                        # fin de ligne
                        return self.hardCodedCommands[1]
                    else :
                        # l'utilisateur décide de deselectionner des elements en plus sur une ligne
                        return self.hardCodedCommands[2]
                elif oldselEnd1 == newselEnd1 and oldselStart1 < newselStart1 :
                    # selectionner des elements sur la ligne
                    return self.hardCodedCommands[2]
                elif isEnd :
                    # deplacement vers la fin de la ligne
                    return self.hardCodedCommands[3]
                else:
                    # déplacement vers la droite
                    return self.hardCodedCommands[4]
        elif oldPos == newPos :
            # rien ne s'est passe
            return None
        # La selection se fait de droite à gauche
        elif oldPos > newPos :
            if oldPos - newPos > 3 :
                # dans le cas où la selection se passe sur la meme ligne (si la ligne est differente, l'utilisateur se deplace vers la droite)  
                if oldselStart1 == newselStart1 and oldselEnd1 > newselEnd1 :
                    if isEnd :
                        # L'utilisateur selectionne/deselectionne la ligne depuis sa position
                        return self.hardCodedCommands[5]
                    else :
                        # l'utilisateur décide de selectionner un element (un mot ou groupe de lettres, comment differencier son intention ?)
                        return self.hardCodedCommands[6]
                elif oldselEnd1 == newselEnd1 and oldselStart1 > newselStart1 :
                    if isStart :
                        # selectionne tout depuis la position vers le debut de la ligne
                        return self.hardCodedCommands[5]
                    else :
                        # selectionne un mot vers la gauche
                        return self.hardCodedCommands[6]

                elif isStart :
                    # deplacement vers le debut de ligne
                    return self.hardCodedCommands[7]
                else :
                    # déplacement vers la gauche
                    return self.hardCodedCommands[8]
    
    def check_cursorEnd(self,cursor):
        # verifie si le curseur se trouve a la fin d'une ligne
        if cursor.columnNumber() == cursor.block().length() - 1 :
            return True
        else :
            return False
        
    
    def check_cursorStart(self,cursor):
        # verifie si le curseur se trouve au debut d'une ligne
        if cursor.columnNumber() == 0 :
            return True
        else :
            return False

        