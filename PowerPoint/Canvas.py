from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from CanvasTools import *
from copy import deepcopy


class Canvas(QWidget):
    def __init__(self, parent = None):
        super(Canvas,self).__init__()
        self.parent = parent
        self.setMinimumSize(300,300)
        self.setMouseTracking(True)
        self.cursorPos = None
        self.pStart = None
        self.setContentsMargins(10,10,10,10)

        # attributs d'affichage
        self.bkcolor = QColor(Qt.blue) # couleur de fond et du contour (par defaut)
        self.width = 3 # taille du contour
        self.painterTranslation = QPoint(0,0) # vecteur de translation
        self.scale = 1 # zoom du canvas

        # attributs mode
        self.mode = 'draw'
        self.currentTool = "drawRect"

        # attributs memoire
        # liste des figures sur le canvas
        self.Lforms = [] # liste de tuple (fonction d'affichage, objet, couleur de fond)
        self.selection = Selection()
        self.alignTool = AlignTool()
        self.copy = None # figures stocké en copy
    

    def addImage(self):
        # tuple of : (QImage, (x,y))
        fileName = QFileDialog.getOpenFileName( self,"Open Image", "./", "Images (*.png *.xpm *.jpg)")[0]
        if len(fileName)==0:
            return 
        self.Lforms.append(QRectImage(QRect(10, 10, 100, 100), QImage(fileName)))
        self.update()

    def mousePressEvent(self, event):
        p = (event.pos() - self.painterTranslation*self.scale)
        self.cursorPos = event.pos()

        # on implémente le resize:
        self.setter = self.selection.resizeSelect(event.pos()/self.scale - self.painterTranslation)
        if self.setter is not None:
            self.mode = 'resize'
            self.pStart = (event.pos()/self.scale - self.painterTranslation)
            return
        
        # On selectionne une figure
        elif self.mode == 'select':
            p = event.pos()/self.scale - self.painterTranslation

            if QApplication.keyboardModifiers() != Qt.ShiftModifier:
                self.selection.clear()
            if not self.selection.toogleSelect(self.Lforms,p): 
                self.selection.clear()
            self.update()
        # On dessine
        else:
            self.pStart = (self.cursorPos/self.scale - self.painterTranslation)
            if self.mode=='draw':
                self.cursorPos = (event.pos()/self.scale - self.painterTranslation)
                if self.currentTool == "drawRect":
                    self.Lforms.append(QRectPlus(QRect(self.pStart, self.pStart), self.bkcolor))
                else:
                    self.Lforms.append(QEllipse(QRect(self.pStart, self.pStart), self.bkcolor))
            self.update()
                            
    def mouseMoveEvent(self, event):
        if self.pStart != None:
            # Si le canvas est deplace, il faut recentre le curseur
            oldV = (self.cursorPos - self.pStart)
            self.cursorPos = event.pos()
            # Pour chaque point on continue de dessiner
            if self.mode=='draw':
                self.cursorPos= (self.cursorPos/self.scale - self.painterTranslation)
                self.Lforms[-1].setBottomRight(self.cursorPos)
                self.update()
            
            # On deplace tout le canvas
            elif self.mode=='move':
                V = (self.cursorPos - self.pStart) - oldV
                self.painterTranslation = self.painterTranslation + V/self.scale
            
            elif self.mode =='resize':
                self.cursorPos= (self.cursorPos/self.scale - self.painterTranslation)
                V = self.cursorPos - self.pStart
                self.setter(V)
                self.pStart = self.cursorPos
            self.update()


    def mouseReleaseEvent(self, event):
        # La figure est dessinee, on l'ajoute dans la liste d'objets
        if self.mode == 'draw':
            pass

        elif self.mode == 'resize':
            self.corner_resize = None
            self.setter = None
            self.mode = 'select'

        self.pStart = None
        self.cursorPos = None
        self.update()

        
    def paintEvent(self, event):
        painter = QPainterPlus(self)
        painter.scale(self.scale, self.scale)
        painter.translate(self.painterTranslation)

        # Toutes les figures
        for form in self.Lforms:
            form.draw(painter)

        self.selection.draw(painter)

    def reset(self):
        print("reset")

    def set_color(self, color):
        # On change la couleur de la figure selectionne ou des prochaines figures
        if self.mode == 'select':
            for o in self.selection.selected:
                o.color = color
            self.update()
        else:
            self.bkcolor = color
    
    # Retourn le canvas en QImage
    def getImage(self):
        size = self.size()
        x, y = size.width(), size.height()
        image = QImage(x, y, QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(image)
        for form in self.Lforms:
            form.draw(painter)

        if self.selection!=None:
            form = self.selection
            pen = QPen(Qt.cyan, 2,  Qt.DashLine)
            painter.setOpacity(0.4)
            form.draw(painter)
        return image


    # On selectionne l'objet en fonction du mode (rectangle et ellipse pour le dessin)
    # On selectionne la nouvelle forme de la figure selectionnee dans le cas select et lasso
    @pyqtSlot()
    def setTool(self,tool):
        if tool == "rectangle":
            self.currentTool = "drawRect"
        elif tool == "ellipse":
            self.currentTool = "drawEllipse"
    
    # On change le mode du canvas
    @pyqtSlot()
    def setMode(self, mode):
        if self.mode != mode:
            self.selection.clear()
            self.mode = mode
            self.update()

    def zoom_in(self):
        self.scale *= 2
        self.update()
    
    def zoom_out(self):        
        self.scale *= 0.5
        self.update()

    @pyqtSlot()
    def copy_element(self):
        if self.mode == "select" and self.selection!=None:
            self.copy = self.selection.copy_contents()

    @pyqtSlot()
    def paste_element(self):
        if len(self.copy) > 0 :
            for o in self.copy: o.translate(20,20)
            self.Lforms.extend(self.copy)
            self.selection.selected = self.copy
            self.copy = [o.copy() for o in self.copy]
            self.update()

    @pyqtSlot()
    def duplicate_element(self):
        self.copy_element()
        self.paste_element()

    def cut_element(self):
        self.copy_element()
        if self.copy!=None:
            self.parent.log_action("Cut "+ self.elementToString(self.copy))
            self.selection = None
            self.update()

    # Supprime le dernier objet sur le canvas
    def deleteSelection(self):
        for o in self.selection.selected:
            self.Lforms.remove(o)
        self.selection.clear()
        self.update()

    # Retourne la chaine de caractere d'un element de l'objet
    def elementToString(self, elt):     
        return ''

    # Pour bouger une figure
    def move_element(self, direction, scale = 1):
        if self.selection!=None:
            v = None
            if direction == 'Up':
                v = QPoint(0,-10)
            elif direction == 'Down':
                v = QPoint(0,10)
            elif direction == 'Left':
                v = QPoint(-10,0)
            elif direction == 'Right':
                v = QPoint(10,0)
            if v != None:
                # Pour les lignes il faut deplacer TOUS les points
                if type(self.selection) is list:
                    for e in range(len(self.selection)):
                        self.selection[e].translate(v)
                    self.update()
                    return
                for o in self.selection.selected:
                    o.translate(v*scale)
                self.update()
    
    def group(self):
        if len(self.selection.selected) > 1:
            G = QRectGroup()
            G.add_objects(self.selection.selected)
            G.update()
            for o in G.objects:
                self.Lforms.remove(o)
            self.Lforms.append(G)
            self.selection.selected = [G]
            self.update()
    
    def ungroup(self):
        if len(self.selection.selected) == 1 and isinstance(self.selection.selected[0], QRectGroup):
            G = self.selection.selected[0]
            for o in G.objects: self.Lforms.append(o)
            self.Lforms.remove(G)
            self.selection.selected = G.objects
            self.update()

    def alignLeft(self):
        if not self.selection.isEmpty():
            self.alignTool.alignLeft(self.selection.selected)
            self.update()

    def alignRight(self):
        if not self.selection.isEmpty():
            self.alignTool.alignRight(self.selection.selected)
            self.update()
    
    def alignTop(self):
        if not self.selection.isEmpty():
            self.alignTool.alignTop(self.selection.selected)
            self.update()

    def alignBottom(self):
        if not self.selection.isEmpty():
            self.alignTool.alignBottom(self.selection.selected)
            self.update()
    