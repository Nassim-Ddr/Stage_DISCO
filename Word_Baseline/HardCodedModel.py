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
    def predict(self, oldState, newState, texteditor):
        #inputData, label = inputData[:len(inputData)-1], inputData[-1]
        #past, present = inputData[:5], inputData[5:]
        past, present = oldState,newState
        oldPos, oldline1, oldcolumn1, oldselStart1, oldselEnd1= past
        newPos, newLine1, newColumn1, newselStart1, newselEnd1 = present
        #print(f'Old selection = {oldselStart1} {oldselEnd1} and new Selection = {newselStart1} {newselEnd1}')

        cursor = texteditor.textCursor()

        # On va plus loin dans le document
        isEnd = self.check_cursorEnd(cursor)
        isStart = self.check_cursorStart(cursor)
        if oldPos < newPos and newPos-oldPos > 3 :      
            if oldselStart1 == newselStart1 and oldselEnd1 < newselEnd1:
                if isEnd :
                    print("CTRL + Shift + Fin (End) Button")
                else :
                    # l'utilisateur décide de deselectionner des elements en plus sur une ligne
                    print("CTRL + Shift + Right")
            elif oldselEnd1 == newselEnd1 and oldselStart1 < newselStart1 :
                print("CTRL + Shift + Right")
            elif isEnd :
                print("CTRL + Fin (End)")
            else:
                # déplacement vers la droite
                print("CTRL + Right")
        elif oldPos == newPos :
            print("rien ne s'est passe ou alors action inverse")
        # La selection se fait de droite à gauche
        elif oldPos > newPos and oldPos - newPos > 3 :
            # dans le cas où la selection se passe sur la meme ligne (si la ligne est differente, l'utilisateur se deplace vers la droite)  
            if oldselStart1 == newselStart1 and oldselEnd1 > newselEnd1 :
                if isEnd :
                    print("CTRL + Shift + Home")
                else :
                    # l'utilisateur décide de selectionner un element (un mot ou groupe de lettres, comment differencier son intention ?)
                    print("CTRL + Shift + Left")
            elif oldselEnd1 == newselEnd1 and oldselStart1 > newselStart1 :
                if isStart :
                    print("CTRL + Shift + Home")
                else :
                    print("CTRL + Shift + Left")

            elif isStart :
                print("CTRL + Home")
            else :
                # déplacement vers la gauche
                print("CTRL + Left")
    
    def check_cursorEnd(self,cursor):

        if cursor.columnNumber() == cursor.block().length() - 1 :
            return True
        else :
            return False
        
    
    def check_cursorStart(self,cursor):

        if cursor.columnNumber() == 0 :
            return True
        else :
            return False

        