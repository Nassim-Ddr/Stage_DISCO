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
        past, present = inputData[:7], inputData[7:]
        oldPos, oldline1, oldcolumn1, oldline2, oldcolumn2, oldselStart1, oldselEnd1= past
        newPos, newLine1, newColumn1, newLine2, newColumn2, newselStart1, newselEnd1 = present



        # On va plus loin dans le document
        if oldPos < newPos and newPos-oldPos > 2 :
            # dans le cas où la selection se passe sur la meme ligne (si la ligne est differente, l'utilisateur se deplace vers la droite)
            if oldline1 == newLine1 :
                # si la colonne est differente ici, l'utilisateur ne fait que se deplacer (vers la droite)
                if oldcolumn1 == newColumn1 :
                    # La fin de la selection se trouve sur la meme ligne
                    if oldline2 == newLine2 :
                        if oldcolumn2 > newColumn2 and oldselEnd1 < newselEnd1 :
                            # l'utilisateur décide de selectionner un element (un mot ou groupe de lettres, comment differencier son intention ?)
                            print("CTRL + Shift + Left")
                        elif oldcolumn2 < newColumn2 and oldselEnd1 > newselEnd1:
                            # l'utilisateur décide de deselectionner des elements en plus sur une ligne
                            print("CTRL + Shift + Right")
                        else:
                            # L'utilisateur s'est déplacé
                            print("CTRL + Right")
                    else :
                        if oldselEnd1 < newselEnd1 :
                            # Une selection sur la meme ligne, donc la fin est plus loin vers la droite dans ce cas
                            print("CTRL + Shift + Right")
                else:
                    print("Deplacement droite (CTRL+Right pour se deplacer de mot en mot)")
            else:
                print("Deplacement droite (CTRL+Right pour se deplacer de mot en mot)")
        elif oldPos == newPos :
            print("rien ne s'est passe")
        # La selection se fait de droite à gauche
        else :
            if oldline2 == newLine2:
                if oldcolumn2 == newColumn2:
                    if oldline1 == newLine1:
                        # old > new -> mouvement vers la gauche
                        if oldcolumn1 > newColumn1 and oldselStart1 > newselStart1 :
                            print("CTRL + Shift + Left")
                        # old < new -> mouvement vers la droite
                        elif oldcolumn1 < newColumn1 and oldselStart1 < newselStart1:
                            print("CTRL + Shift + Right")
                        else:
                            print("CTRL + Left")
                    else :
                        # si on passe a la ligne precedente, on sait que c'est une selection de droite a gauche
                        print("CTRL + Shift + Left")
                else:
                    print("Deplacement gauche (CTRL+Left pour se deplacer de mot en mot)")
            else :
                print("Deplacement gauche (CTRL+Left pour se deplacer de mot en mot)")
        