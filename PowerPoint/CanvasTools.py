from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import QRect, QPoint, Qt
import numpy as np


class QRectPlus(QRect):    
    def __init__(self, rect, color=Qt.blue, border_color = Qt.blue, border_width= 5):
        QRect.__init__(self, rect)
        self.color = color
        self.border_color = border_color
        self.border_width = border_width
    
    
    def draw(self, painter):
        painter.setPen(QPen(self.border_color, self.border_width))
        painter.setBrush(self.color)
        painter.drawRect(self)
    
    def adjustTopLeft(self, c):
        self.setTopLeft(self.topLeft() + c)
    
    def adjustTopRight(self, c):
        self.setTopRight(self.topRight() + c)
    
    def adjustBottomLeft(self, c):
        self.setBottomLeft(self.bottomLeft() + c)
        
    def adjustBottomRight(self, c):
        self.setBottomRight(self.bottomRight() + c)
    
    def copy(self):
        return QRectPlus(self, self.color, self.border_color, self.border_width)
    
    # Compare form
    # mode: 'all', 'style', 'color'
    def equals(self, eps=0, mode = 'all'):
        return True
    
    def __str__(self):
        return f"{self.__class__.__name__}{self.top(),self.left(), self.height(), self.width()}"
    
    def __repr__(self):
        return self.__str__()

class QEllipse(QRectPlus):
    def draw(self, painter):
        painter.setPen(QPen(self.border_color, self.border_width))
        painter.setBrush(self.color)
        painter.drawEllipse(self)

    def copy(self):
        return QEllipse(self, self.color, self.border_color, self.border_width)
    
class QTriangle(QRectPlus):
    def draw(self, painter):
        painter.setPen(QPen(self.border_color, self.border_width))
        painter.setBrush(self.color)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.moveTo(self.bottomLeft())
        path.lineTo(self.bottomRight())
        path.lineTo(self.center().x(),self.top())
        path.lineTo(self.bottomLeft())
        painter.drawPath(path)

    def copy(self):
        return QTriangle(self, self.color, self.border_color, self.border_width)


# QRect with a image inside the rectangle
class QRectImage(QRectPlus):
    def __init__(self, rect, image):
        QRectPlus.__init__(self, rect)
        self.image = image
    
    def draw(self, painter):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"]
        L = [(getattr(self, f)().x(), getattr(self, f)().y()) for f in Lf]
        x1,y1 = np.min(L,0)
        x2,y2 = np.max(L,0)
        print(self)
        painter.drawImage(QRect(x1,y1,x2-x1,y2-y1), self.image)
    
    def copy(self):
        return QRectImage(self, self.image)


class QRectGroup(QRect):
    def __init__(self, objects = []):
        QRect.__init__(self, 0,0,10,10)
        self.objects = objects
        if objects:
            self.update()
    
    def translate(self, v):
        super().translate(v)
        for o in self.objects:
            o.translate(v)

    def add_object(self, o):
        self.objects.append(o)
        self.update()
    
    def add_objects(self, objects):
        self.objects = objects
        self.update()

    def remove_object(self, o):
        self.objects.remove(o)
        self.update()
    
    def update(self):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"] 
        L = [(getattr(o, f)().x(), getattr(o, f)().y()) for f in Lf for o in self.objects]
        x1,y1 = np.min(L,0)
        x2,y2 = np.max(L,0)
        self.setTopLeft(QPoint(x1, y1))
        self.setBottomRight(QPoint(x2, y2))

    def adjust(self, s, c):
        for o in self.objects:
           getattr(o, s)(c)
        self.update()
    
    def adjustBottomRight(self, c):
        self.adjust("adjustBottomRight", c)
    
    def adjustBottomLeft(self, c):
        self.adjust("adjustBottomLeft", c)
    
    def adjustTopRight(self, c):
        self.adjust("adjustTopRight", c)
    
    def adjustTopLeft(self, c):
        self.adjust("adjustTopLeft", c)

    def draw(self, painter):
        for o in self.objects:
            o.draw(painter)
    
    def copy(self):
        G = QRectGroup()
        G.add_objects( [o.copy() for o in self.objects] )
        return G


# My QPainter
class QPainterPlus(QPainter):
    def drawRectImage(self, rect):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"]
        L = [(getattr(rect, f)().x(), getattr(rect, f)().y()) for f in Lf]
        x1,y1 = np.min(L,0)
        x2,y2 = np.max(L,0)
        self.drawImage(QRect(x1,y1,x2-x1,y2-y1), rect.image)
    

class Selection():
    def __init__(self):
        self.selected = []
        self.element = None

    def clear(self):
        self.selected.clear()
    
    # add element in selection
    def add_element(self, f):
        if f is not None and f not in self.selected:
            self.selected.append(f)
            return True
        return False
    
    # add element in selection
    def remove_element(self, f):
        try:
            self.selected.remove(f)
        except:
            pass

    # find a object containing the cursor
    def find(self, board, cursor):
        for i in range(len(board)-1, -1, -1):  
            if board[i].contains(cursor): return board[i]
        return None

    def contains(self, f):
        for o in self.selected:
            if o == f: return True
        return False
        
    def resizeSelect(self, cursor):
        for o in self.selected:
            Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"]
            Lsetter = ["adjustTopLeft", "adjustTopRight", "adjustBottomLeft", "adjustBottomRight"]
            for f,setter in zip(Lf,Lsetter):
                p = getattr(o, f)()
                if (cursor - p).manhattanLength() < 15:
                    print(setter)
                    return getattr(o, setter)
        return None
            
    def draw(self, painter):
        for o in self.selected:
            # carré de selection
            painter.setPen(QPen(Qt.black, 2,  Qt.DashLine))
            painter.setOpacity(0.5)
            painter.setBrush(Qt.transparent)
            painter.drawRect(o)

            # points de resize
            L = [getattr(o, f)() for f in ["topLeft", "topRight", "bottomLeft", "bottomRight"]]
            painter.setPen(QPen(Qt.gray, 10))
            painter.drawPoints(L)
    
    def copy_contents(self):
        copy = [o.copy() for o in self.selected]
        return copy

    def isEmpty(self):
        return len(self.selected) == 0


class AlignTool():
    def alignTop(self, objects):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"] 
        L = [getattr(o, f)().y() for f in Lf for o in objects]
        y = np.min(L)
        for o in objects:
            o.translate(-QPoint(0,min(o.top()-y, o.bottom()-y)))
    
    def alignBottom(self, objects):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"] 
        L = [getattr(o, f)().y() for f in Lf for o in objects]
        y = np.max(L)
        for o in objects:
            o.translate(QPoint(0,min(y-o.top(), y-o.bottom())))

    def alignLeft(self, objects):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"] 
        L = [getattr(o, f)().x() for f in Lf for o in objects]
        x = np.min(L)
        for o in objects:
            o.translate(-QPoint(min(o.right()-x, o.left()-x), 0))
    
    def alignRight(self, objects):
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"] 
        L = [getattr(o, f)().x() for f in Lf for o in objects]
        x = np.max(L)
        for o in objects:
            o.translate(QPoint(min(x-o.right(), x-o.left()),0))

        
        



