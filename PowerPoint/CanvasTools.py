from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np

class QRectImage(QRect):
    def __init__(self,x,y, w, h, image):
        QRect.__init__(self,x,y,w,h)
        self.image = image

class QPainterPlus(QPainter):
    def drawRectImage(self, rect):  
        self.drawImage(rect, rect.image)

