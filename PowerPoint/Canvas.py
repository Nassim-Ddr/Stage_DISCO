from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from CanvasTools import *


class Canvas(QWidget):
    def __init__(self, parent = None):
        super(Canvas,self).__init__()
        self.parent = parent
        self.setMinimumSize(300,300)
        self.setMouseTracking(True)
        self.cursorPos = None
        self.pStart = None
        self.setStyleSheet("background-color: red")
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
        self.selected = None # figure selectionnée
        self.copy = None # figure stocké en copy
        self.lasso = [] # les traits du lasso
        self.polygon = None # le polygon du lasso
        self.intersectionPoint = None # le point d'intersection du scriboli
        self.pictab = [] # Liste d'images
    
    def addImage(self):
        #print("adding pic")
        # tuple of : (QImage, (x,y))

        fileName = QFileDialog.getOpenFileName( self,"Open Image", "/home", "Images (*.png *.xpm *.jpg)")[0]
        if len(fileName)==0:
            return 
        affiche, form, c = "drawRectImage", QRectImage(10, 10, 100, 100, QImage(fileName)), Qt.red
        self.Lforms.append((affiche, form, c))
        self.update()
    
        


    def resize(self, cursor):
        if self.selected is None: return False
        Lf = ["topLeft", "topRight", "bottomLeft", "bottomRight"]
        Lsetter = ["setTopLeft", "setTopRight", "setBottomLeft", "setBottomRight"]
        for f,setter in zip(Lf,Lsetter):
            p = getattr(self.selected[1], f)()
            if (cursor.pos() - p).manhattanLength() < 15:
                self.corner_resize = p 
                self.setter = setter
                self.mode = 'resize'
        return False

    def mousePressEvent(self, event):
        p = (event.pos() - self.painterTranslation*self.scale)
        self.cursorPos = event.pos()

        # on implémente le resize:
        if self.resize(event):
            return

        # On implemente le lasso
        if  self.mode == 'lasso':
            self.pStart = p
            self.lasso.append(p)

        # On implemente le scriboli
        elif self.mode == 'scriboli':
            self.pStart = p
            self.cursorPos = p
        
        # On selectionne une figure
        elif self.mode == 'select':
            p = event.pos()/self.scale - self.painterTranslation
            for i in range(len(self.Lforms)-1, -1, -1):
                # Freedrawing case
                if self.Lforms[i][0] == 'drawLines':
                    for e in self.Lforms[i][1]:
                        p1 = (p - e.p1()) 
                        if p1.x() < 8 and p1.y() < 8:
                            self.select(self.Lforms.pop(i))
                            self.update()
                            return
                # Qrects (ellipses and rectangles)        
                elif self.Lforms[i][1].contains(p):
                    self.select(self.Lforms.pop(i))
                    self.update()
                    return
            self.select(None)
            self.update()
        
        # On dessine
        else:
            self.pStart = (self.cursorPos/self.scale - self.painterTranslation)
            if self.mode=='draw':
                self.cursorPos = (event.pos()/self.scale - self.painterTranslation)
                # Dessin Libre
                if self.currentTool == 'drawLines':
                    lines = [QLineF(self.pStart, self.pStart)]
                    self.Lforms.append([self.currentTool, lines, self.bkcolor])
                    self.update()
                    return
                # Autres formes: rectangle et ellipse
                rect = QRect(self.pStart.x(), self.pStart.y(), 0, 0)
                self.Lforms.append([self.currentTool, rect, self.bkcolor])
        self.update()
                            
    def mouseMoveEvent(self, event):
        if self.pStart != None:
            # Si le canvas est deplace, il faut recentre le curseur
            oldV = (self.cursorPos - self.pStart)
            self.cursorPos = event.pos()
            # Pour chaque point on continue de dessiner
            if self.mode=='draw':
                self.cursorPos= (self.cursorPos/self.scale - self.painterTranslation)
                if self.currentTool == 'drawLines':
                    self.Lforms[-1][1].append(QLineF(self.pStart, self.cursorPos))
                    self.pStart = self.cursorPos
                else:
                    self.Lforms[-1][1].setBottomRight(self.cursorPos)
            
            # Pour chaque point de la ligne, on dessine ce point jusqu'a ce qu'on fasse 1 noeud (le dernier noeud sera celui pris en compte pour le scriboli)
            elif self.mode =='scriboli':
                # Si le canvas est deplace, il faut recentre le curseur
                self.cursorPos-= self.painterTranslation
                line = QLineF(self.pStart, self.cursorPos)
                self.pStart = self.cursorPos
                self.lasso.append(line)
                for l in self.lasso[:-10]:
                    # On cherche une queue de cochon
                    if line.intersects(l)[0] == 1:
                        # On utilise QPolygon pour attraper l'objet (unique) dans le cercle.
                        polygon = [l.p1().toPoint() for l in self.lasso]
                        polygon = QPolygon(polygon)
                        self.intersectionPoint = line.p1()
                        # On cherche la figure potentiellement dans le dessin
                        for i in range(len(self.Lforms)-1, -1, -1):
                            rect = self.Lforms[i][1]
                            # La figure est dedans ?
                            if self.polygonContains(polygon, rect):
                                # OUI !
                                self.select(self.Lforms.pop(i))
                                self.update()       
                self.update()
            
            # On dessine le polygone de selection tant qu'on maintient la souris, on garde la premiere forme trouvee
            elif self.mode == 'lasso':
                # Si le canvas est deplace, il faut recentre le curseur
                self.cursorPos-= self.painterTranslation
                self.lasso.append(self.cursorPos)
                self.polygon = QPolygon(self.lasso)
                # Si on attrape une figure tant qu'ne lache pas on ne fait rien
                if self.selected != None and self.polygonContains(self.polygon, self.selected[1]):
                    self.update()
                    return
                
                # On cherche une figure qui est attrapee
                for i in range(len(self.Lforms)-1, -1, -1):
                    rect = self.Lforms[i][1]
                    if self.polygonContains(self.polygon, rect):
                        self.select(self.Lforms.pop(i))
                        self.update()
                        return
                    
                # On relance un lasso
                self.select(None)
            
            # On deplace tout le canvas
            elif self.mode=='move':
                V = (self.cursorPos - self.pStart) - oldV
                self.painterTranslation = self.painterTranslation + V/self.scale
            
            elif self.mode =='resize':
                self.cursorPos= (self.cursorPos/self.scale - self.painterTranslation)
                getattr(self.selected[1], self.setter)(self.cursorPos)

            self.update()


    def mouseReleaseEvent(self, event):
        # La figure est dessinee, on l'ajoute dans la liste d'objets
        if self.mode == 'draw':
            self.add_object()

        elif self.mode == 'resize':
            self.corner_resize = None
            self.setter = None
            self.mode = 'selected'

        # Si on lache en mode lasso ou scriboli, on libere le lasso et le polygone et en mode scriboli, en fonction de l'angle quelque chose se passe pour les figures.
        elif self.mode == 'lasso' or 'scriboli':
            if self.intersectionPoint != None and self.selected != None:
                finalpos = self.cursorPos
                ligne = QLineF(self.intersectionPoint,finalpos)

                # pour detecter l'angle la plus proche
                angle = ligne.angle()
                haut = abs(90-angle)
                droite = min(abs(angle), abs(angle-360))
                gauche = abs(180-angle)
                bas = abs(270-angle)

                L = [Qt.black, Qt.red, Qt.magenta, Qt.green]
                angles = [haut, gauche, droite, bas]
                self.selected[2] = QColor(L[np.argmin(angles)]) # on cherche l'angle la plus proche
                self.select(None)

            self.intersectionPoint = None
            self.lasso = []
            self.polygon = None

        self.pStart = None
        self.cursorPos = None
        self.update()

        
    def paintEvent(self, event):
        painter = QPainterPlus(self)
        painter.setBackground(QBrush(Qt.red))
        painter.scale(self.scale, self.scale)
        painter.translate(self.painterTranslation)

        # Toutes les figures
        for affiche, form, c in self.Lforms:
            painter.setPen(QPen(c, self.width))
            painter.setBrush(c)
            getattr(painter, affiche)(form)

        # On affiche une figure fantome 
        if self.selected!=None:
            # carré de selction
            painter.setPen(QPen(Qt.black, 1,  Qt.DashLine))
            painter.setOpacity(0.2)
            affiche, form, c = self.selected
            painter.drawRect(form)
            L = [getattr(form, f)() for f in ["topLeft", "topRight", "bottomLeft", "bottomRight"]]
            painter.setPen(QPen(Qt.black, 10,  Qt.DashLine))
            painter.drawPoints(L)


            # objet sélectionné
            painter.setPen(QPen(Qt.black, 2))
            painter.setOpacity(0.5)
            painter.setBrush(c)
            getattr(painter, affiche)(form)

        
        # Dessine le lasso
        if self.polygon!=None:
            painter.setBrush(Qt.blue)
            painter.setOpacity(0.1)
            painter.drawPolygon(self.polygon)

        # Dessine la ligne du scriboli
        if self.mode =='scriboli':
            painter.setPen(QPen(Qt.red, 2))
            painter.setOpacity(1)
            painter.drawLines(self.lasso)
        
        


    # Fonction intermediaire pour verifier si une figure se trouve dans un polygone
    # On verifie que si un des 4 coins du cadre de la figure est dedans
    def polygonContains(self, polygon, figure):
        if type(figure) is list: # si la figure est un trait
            for l in figure:
                if polygon.containsPoint(l.p1().toPoint(),Qt.OddEvenFill):
                    return True
            return False
        rect = figure
        x1 = polygon.containsPoint(rect.topLeft(), Qt.OddEvenFill)
        x2 = polygon.containsPoint(rect.bottomLeft(), Qt.OddEvenFill)
        x3 = polygon.containsPoint(rect.topRight(), Qt.OddEvenFill)
        x4 =  polygon.containsPoint(rect.bottomRight(), Qt.OddEvenFill)
        return x1 or x2 or x3 or x4

    # Change la selection
    def select(self, figure):
        if self.selected != None:
            self.Lforms.append(self.selected)
        self.selected = figure

    def reset(self):
        print("reset")

    # ajoute dans les logs, l'evenement d'ajout d'un element
    def add_object(self):
        self.parent.log_action("added - "+ self.elementToString(self.Lforms[-1]))

    def set_color(self, color):
        # On change la couleur de la figure selectionne ou des prochaines figures
        if (self.mode == 'select' or self.mode == 'lasso') and self.selected!=None:
            if self.selected[1]!=color:
                self.selected[2] = color
                self.parent.log_action(self.elementToString(self.selected) + ' - change color into {}'.format(color.getRgb()))
                self.update()
        else:
            self.bkcolor = color
    
    # Retourn le canvas en QImage
    def getImage(self):
        size = self.size()
        x, y = size.width(), size.height()
        image = QImage(x, y, QImage.Format_ARGB32_Premultiplied)
        painter = QPainter(image)
        for affiche, form, c in self.Lforms:
            painter.setPen(QPen(c, self.width))
            painter.setBrush(c)
            getattr(painter, affiche)(form)

        if self.selected!=None:
            affiche, form, c = self.selected
            pen = QPen(Qt.cyan, 2,  Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(c)
            painter.setOpacity(0.4)
            getattr(painter, affiche)(form)
        return image


    # On selectionne l'objet en fonction du mode (rectangle et ellipse pour le dessin)
    # On selectionne la nouvelle forme de la figure selectionnee dans le cas select et lasso
    @pyqtSlot()
    def setTool(self,tool):
        if (self.mode == 'select' or self.mode == 'lasso') and self.selected!=None:
            if self.selected[0] == 'drawRect' and tool!='ellipse':
                self.selected[0] = 'drawEllipse'
                self.parent.log_action(self.elementToString(self.selected) + ' -  change form into ellipse')
            elif self.selected[0] == 'drawEllipse' and tool=='rectangle':
                self.selected[0] = 'drawRect'
                self.parent.log_action(self.elementToString(self.selected) + ' -  change form into rectangle')
            self.update()
        else:
            if tool == "rectangle":
                self.currentTool = "drawRect"
            elif tool == "ellipse":
                self.currentTool = "drawEllipse"
            elif tool == "drawLines":
                self.currentTool = "drawLines"
    
    # On change le mode du canvas
    @pyqtSlot()
    def setMode(self, mode):
        if self.mode != mode:
            if self.lasso!=None:
                self.lasso = []
                self.polygon = None
            elif self.selected!=None:
                self.select(None)
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
        if self.mode == "select" and self.selected!=None:
            self.copy = self.selected
            self.parent.log_action("")

    @pyqtSlot()
    def paste_element(self):
        if self.copy!=None:
            affiche, form, c = self.copy
            if affiche == 'drawLines':
                return
            self.copy = affiche, form.translated(20, 20), c
            self.Lforms.append(self.copy)
            self.update()

    @pyqtSlot()
    def duplicate_element(self):
        self.copy_element()
        if self.copy!=None:
            if self.copy[0]== 'drawLines':
                return
            self.parent.log_action("Duplicate "+ self.elementToString(self.copy))
            self.paste_element()
            self.copy = None

    def cut_element(self):
        self.copy_element()
        if self.copy!=None:
            if self.copy[0]=='drawLines':
                return
            self.parent.log_action("Cut "+ self.elementToString(self.copy))
            self.selected = None
            self.update()

    # Supprime le dernier objet sur le canvas
    def deleteLastObject(self):
        if self.selected!=None:
            self.selected = None
        elif len(self.Lforms) == 0: 
            return
        else: 
            self.Lforms.pop()
        self.update()

    # Retourne la chaine de caractere d'un element de l'objet
    def elementToString(self, elt):     
        affiche, form, c = elt

        s = ''
        if affiche == 'drawLines': 
            return 'form: lines, color: {}'.format(c.getRgb())
        elif affiche == 'drawRect': s = 'rectangle '
        else: s = 'ellipse '
        return 'form: {}, dimension: {},  color: {}'.format(s, form.getRect(), c.getRgb())

    # Pour bouger une figure
    def move_element(self, direction, scale = 1):
        if self.selected!=None:
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
                if type(self.selected[1]) is list:
                    for e in range(len(self.selected[1])):
                        self.selected[1][e].translate(v)
                    self.update()
                    return
                self.selected[1].translate(v*scale)
                self.update()