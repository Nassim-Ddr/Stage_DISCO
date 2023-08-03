import csv
import numpy as np
import sys
from PIL import Image
import Canvas
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os

class Logger():
    def __init__(self, write = False, recommender = None):
        # Variable pour l'ecriture du fichier
        self.write = write

        # autres variables
        self.prevState = None
        self.cpt = 0 
        self.recommender = recommender
        
    def update(self, state, command, autre=None):
        if self.recommender is not None:
            self.recommender.update(state, autre)
            return
        if self.write:
            image = self.getImage(state)
            image.save(f'./data/{command}/{self.cpt}.jpg')
            self.cpt+= 1
        self.prevState = state
        
    
    def getImage(self, image):
        image1 = self.prevState
        image2 = image
        size = image1.size() + image2.size()

        res = QImage(size.width(), size.height(), QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(res)
        painter.drawImage(0,0, image1)
        painter.drawImage(image1.width(),0, image2)
        painter.end()
        return res

class Player():
    # parameters:
    # nb of initial state
    # nb of actions per initial state
    def start(self, actions, start_funtion=lambda: print(None), reset_function=lambda: print(None), call_function=lambda: print(None), parameters=(1000, 1)):
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for a in actions:
            if not os.path.exists(f'data/{a.__name__}'):
                os.makedirs(f'data/{a.__name__}')


        iter_max, nb_act = parameters 
        i = 0
        while i < iter_max:
            j = 0
            start_funtion()
            while j < nb_act:
                act = np.random.choice(actions)
                call_function()
                act()
                j+=1
            reset_function()
            i+=1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    canvas = Canvas.Canvas()
    actions = [
        canvas.alignBottom,
        canvas.alignLeft,
        canvas.alignRight,
        canvas.alignTop
    ]
    def call_function():
        n = np.random.randint(2,len(canvas.Lforms)+1)
        canvas.selection.selected = np.random.choice(canvas.Lforms, size = n, replace=False)
    canvas.logger.write = True
    player = Player()
    player.start(actions, canvas.randomize, canvas.reset, call_function, parameters=(10,1))
        






        
