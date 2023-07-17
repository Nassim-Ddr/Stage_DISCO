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


class HardCodedModel():
    def predict(self, inputData):
        inputData, label = inputData[:len(inputData)-1], inputData[-1]
        past, present = inputData[:5], inputData[5:]
        cursorPos, line1, column1, line2, column2 = past
        presentPos, presLine1, presColumn1, presLine2, presColumn2 = present

        # On etait plus loin dans le document
        if cursorPos < presentPos and presentPos-cursorPos > 2 :
            # dans le cas où la selection se passe sur la meme ligne
            if line1 == presLine1 :
                if line2 == presLine2 :
                    if column1 == presColumn1 :
                        if column2 < presColumn2 :
                            # l'utilisateur décide de deselectionner un element
                            print("CTRL + Shift + Left")
                        elif column2 > presColumn2 :
                            # l'utilisateur décide de selectionner des elements en plus sur une ligne
                            print("CTRL + Shift + Right")
                        else:
                            # L'utilisateur s'est déplacé
                            print("CTRL + Right")

            

        # c'est juste un deplacement vers la droite il n'y a rien à suggérer
        else :
            print("Right")