import csv
import numpy as np
import sys
from PIL import Image
import Canvas
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Logger():
    def __init__(self, filename = None):
        # Variable pour l'ecriture du fichier
        self.file = filename
        self.writer = None
        if self.file is not None:
            self.file = open(self.file, 'w', newline='')
            self.writer = csv.writer(self.file)

        # autres variables
        self.prevState = None
        self.cpt = 0 
        
    def update(self, state, command):
        if self.file is not None:
            image = self.getImage(state)
            image.save(f'./data/{self.cpt}.jpg')
            self.writer.writerow([self.cpt, command])
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
        canvas.selection.selected = canvas.Lforms
    player = Player()
    player.start(actions, canvas.randomize, canvas.reset, call_function, parameters=(10,1))
    canvas.logger.file.close()
        






        
