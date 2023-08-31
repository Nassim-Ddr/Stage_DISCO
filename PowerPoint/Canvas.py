from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from PyQt5.QtCore import QRect, QPoint, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPixmap
import numpy as np
from CanvasTools import *
from Logger import Logger


class Canvas(QWidget):
    def __init__(self, parent = None):
        super(Canvas,self).__init__()
        self.parent = parent
        # configurations du canvas
        self.setMinimumSize(600,300)
        
        # attributs d'affichage
        self.bkcolor = QColor(Qt.blue)
        self.border_color = QColor(Qt.blue)
        self.width = 3 # taille du contour
        self.painterTranslation = QPoint(0,0) # vecteur de translation
        self.scale = 1 # zoom du canvas

        # attributs mode
        self.mode = 'draw'
        self.currentTool = "drawRect"

        # attributs memoire
        self.Lforms = []
        self.selection = Selection()
        self.alignTool = AlignTool()
        self.copy = None # figures stocké en copy
        self.copyAlign = None

        self.cursorPos = None
        self.pStart = None
        self.command = None
        
        # Logger
        self.logger = Logger('data/data.csv')
    
    def addImage(self):
        # tuple of : (QImage, (x,y))
        fileName = QFileDialog.getOpenFileName( self,"Open Image", "./", "Images (*.png *.xpm *.jpg)")[0]
        if len(fileName)==0:
            return 
        print(fileName)
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
            f = self.selection.find(self.Lforms, p)
            modifiers = QApplication.keyboardModifiers()
            # Séléction Multiple
            if modifiers == Qt.ShiftModifier:
                if f is None: pass
                elif self.selection.isEmpty(): self.selection.add_element(f)
                elif not self.selection.contains(f):
                    self.selection.add_element(f)
                else:
                    self.selection.remove_element(f)
            # Sélection: Copy + Align / Copy + Drag
            elif  modifiers == Qt.ControlModifier or  modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
                if f is None: pass
                elif not self.selection.contains(f):
                    self.selection.add_element(f)
                    self.selection.element = None
                else:
                    self.selection.element = f
                self.copy_element(mode='align')
                self.update()
                if modifiers == (Qt.ControlModifier | Qt.ShiftModifier): self.command = "Copy + Align"
                elif modifiers == Qt.ControlModifier: self.command = "Copy + Drag"
            else:    # Sélection normale
                if f is None: self.selection.clear()
                elif self.selection.isEmpty(): self.selection.add_element(f)
                elif not self.selection.contains(f):
                    self.selection.clear()
                    self.selection.add_element(f)

            self.pStart = (event.pos()/self.scale - self.painterTranslation)
        # On dessine
        else:
            self.pStart = (self.cursorPos/self.scale - self.painterTranslation)
            if self.mode=='draw':
                self.cursorPos = (event.pos()/self.scale - self.painterTranslation)
                args = (QRect(self.pStart, self.pStart), self.bkcolor, self.border_color, self.width)
                match self.currentTool:
                    case "drawRect":
                        self.Lforms.append(QRectPlus(*args))
                    case "drawTriangle":
                        self.Lforms.append(QTriangle(*args))
                    case other:
                        self.Lforms.append(QEllipse(*args))
            self.update()
                            
    def mouseMoveEvent(self, event):
        if self.pStart is not None:
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
            
            elif self.mode == 'select':
                modifiers = QApplication.keyboardModifiers()
                self.paste_element((0,0), 'align')
                self.selection.element = None
                self.cursorPos= (self.cursorPos/self.scale - self.painterTranslation)
                # AlignCopy move
                if modifiers == (Qt.ControlModifier | Qt.ShiftModifier): 
                    V = self.cursorPos - self.pStart
                    V.setY(0) if np.abs(V.x()) > np.abs(V.y()) else V.setX(0)
                    oldV.setY(0) if np.abs(oldV.x()) > np.abs(oldV.y()) else oldV.setX(0)
                    for o in self.selection.selected: o.translate(-oldV)
                    for o in self.selection.selected: o.translate(V)
                else: # Normal move
                    V = self.cursorPos - self.pStart
                    for o in self.selection.selected: o.translate(V)
                    self.pStart = self.cursorPos
                    self.command = "Move"
            self.update()

    def updated(self, command):
        self.logger.update(self.getImage(), command, self.state())
    
    def state(self):
        return [o.copy() for o in self.Lforms]

    def mouseReleaseEvent(self, event):
        # La figure est dessinee, on l'ajoute dans la liste d'objets
        if self.mode == 'draw':
            self.updated('draw')
        elif self.mode == 'resize':
            self.corner_resize = None
            self.setter = None
            self.mode = 'select'
            self.updated('Resize')
        elif self.mode == 'select':
            self.copyAlign = None
            self.selection.remove_element(self.selection.element)
            if self.command is not None:
                self.updated(self.command)
                self.command = None
        self.pStart = None
        self.cursorPos = None
        self.command = None
        self.update()
        

        
    def paintEvent(self, event):
        painter = QPainterPlus(self)
        painter.scale(self.scale, self.scale)
        painter.translate(self.painterTranslation)

        # Toutes les figures
        for form in self.Lforms: form.draw(painter)
        self.selection.draw(painter)
        if self.pStart is None: self.drawPreview()
    
    def drawPreview(self):
        image = self.getImage()
        image = image.scaled(self.parent.preview.size())
        self.parent.preview.setPixmap(QPixmap.fromImage(image))

    def reset(self):
        # attributs d'affichage
        self.bkcolor = QColor(Qt.blue) # couleur de fond et du contour (par defaut)
        self.width = 3 # taille du contour
        self.painterTranslation = QPoint(0,0) # vecteur de translation
        self.scale = 1 # zoom du canvas

        # attributs mode
        self.mode = 'draw'
        self.currentTool = "drawRect"

        # attributs memoire
        self.Lforms = []
        self.selection = Selection()
        self.alignTool = AlignTool()
        self.copy = None # figures stocké en copy
        self.cursorPos = None
        self.pStart = None

        self.logger.prevState = None

    def set_color(self, color):
        # On change la couleur de la figure selectionne ou des prochaines figures
        if self.mode == 'select' and not self.selection.isEmpty():
            for o in self.selection.selected: o.color = color
            self.update()
            self.updated('Change color')
        else:
            self.bkcolor = color

    def set_color_border(self, color):
        # On change la couleur de la figure selectionne ou des prochaines figures
        if self.mode == 'select' and not self.selection.isEmpty():
            for o in self.selection.selected: o.border_color = color
            self.update()
            self.updated('Change border color')
        else:
            self.border_color = color

    
    # Retourn le canvas en QImage
    def getImage(self):
        size = self.size()
        x, y = size.width(), size.height()
        image = QImage(x, y, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.white)
        painter = QPainterPlus(image)
        painter.scale(self.scale, self.scale)
        painter.translate(self.painterTranslation)
        # Toutes les figures
        for form in self.Lforms:
            form.draw(painter)
        painter.end()
        return image


    # On selectionne l'objet en fonction du mode (rectangle et ellipse pour le dessin)
    # On selectionne la nouvelle forme de la figure selectionnee dans le cas select et lasso
    @pyqtSlot()
    def setTool(self,tool):
        if tool == "rectangle":
            self.currentTool = "drawRect"
        elif tool == "ellipse":
            self.currentTool = "drawEllipse"
        elif tool == 'triangle':
            self.currentTool = "drawTriangle"
    
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
    def copy_element(self, mode = 'normal'):
        if self.mode == "select" and not self.selection.isEmpty():
            if mode == 'align':
                self.copyAlign = self.selection.copy_contents()
            else:
                self.copy = self.selection.copy_contents()

    @pyqtSlot()
    def paste_element(self, vector = (20,20), mode = 'normal'):
        vector = QPoint(*vector)
        if mode == 'align':
            if self.copyAlign is not None and len(self.copyAlign) > 0 :
                self.Lforms.extend(self.copyAlign)
                self.selection.selected = self.copyAlign
                self.copyAlign = None
                self.update()
        elif self.copy is not None and len(self.copy) > 0 :
                for o in self.copy: o.translate(vector)
                self.Lforms.extend(self.copy)
                self.selection.selected = self.copy
                self.copy = [o.copy() for o in self.copy]
                self.update()

    @pyqtSlot()
    def duplicate_element(self):
        self.copy_element()
        self.paste_element()
        self.copy = None

    def cut_element(self):
        self.copy_element()
        if self.copy!=None:
            for o in self.selection.selected:
                self.Lforms.remove(o)
            self.selection.clear()
            self.update()

    # Supprime le dernier objet sur le canvas
    def deleteSelection(self):
        for o in self.selection.selected:
            self.Lforms.remove(o)
        self.selection.clear()
        self.update()
        self.updated("Delete")

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
            self.updated('AlignLeft')

    def alignRight(self):
        if not self.selection.isEmpty():
            self.alignTool.alignRight(self.selection.selected)
            self.update()
            self.updated('AlignRight')
    
    def alignTop(self):
        if not self.selection.isEmpty():
            self.alignTool.alignTop(self.selection.selected)
            self.update()
            self.updated('AlignTop')

    def alignBottom(self):
        if not self.selection.isEmpty():
            self.alignTool.alignBottom(self.selection.selected)
            self.update()
            self.updated('AlignBottom')
    
    def randomize(self):
        # create object not inside another object
        def randomObject(L):
            figure  = None
            for _ in range(50):
                inside = False
                x1,y1, x2,y2 = np.random.randint(0,300, size=4)
                c1 = QColor(*np.random.randint(0, 255, size=3))
                c2 = QColor(*np.random.randint(0, 255, size=3))
                args = (QRect(QPoint(x1,y1), QPoint(x2,y2)), c1, c2)
                figure = QRectPlus(*args) if rand==0 else QEllipse(*args)
                for r in L:
                    if figure.contains(r) or r.contains(figure):
                        inside = True
                        break
                if not inside: 
                    break
            return figure


        number = np.random.randint(2, 5)
        objects = []
        for rand in np.random.binomial(1, 0.5, number):
            objects.append(randomObject(objects))
        self.Lforms = objects
        self.selection.clear()
        self.update()
        self.logger.prevState = self.getImage()

    def deplaceLast(self):
        if not self.selection.isEmpty():
            for form in self.selection.selected:
                self.Lforms.remove(form)
                self.Lforms.append(form)
            self.update()
            self.updated("Premier Plan")

    def deplaceFirst(self):
        if not self.selection.isEmpty():
            for form in self.selection.selected:
                self.Lforms.remove(form)
                self.Lforms.insert(0, form)
            self.update()
            self.updated("Arriere Plan")

        
        