import sys
import time

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.color_text = "blue"
        self.container = QWidget()
        layout = QVBoxLayout( self.container )
        
        # A Qwidget embedding a matplotlib figure
        self.canvas  = FigureCanvas( plt.figure( figsize=(5, 3)) )
        layout.addWidget( self.canvas )
        
        # add the matplotlib toolbar
        self.addToolBar( NavigationToolbar( self.canvas, self ) )
        
        self.button = QPushButton( "red" )
        self.button.clicked.connect( self.update_canvas )
        layout.addWidget( self.button )

        self.ax = self.canvas.figure.subplots()
        
        self.update_canvas()
        
        self.setCentralWidget( self.container )

    def update_canvas(self):
        self.count += 1
        
        self.ax.clear()
        t = np.linspace( 0, 10, 101 )
        self.ax.plot( t, np.sin(t + self.count ), c= self.color_text )
        self.ax.set_xlabel( "Time (s)" )
        self.ax.set_ylabel( "Temperature (celcius)" )
        self.canvas.draw()
        self.canvas.repaint()  # required for MACOS
        
        self.color_text = "blue" if self.color_text == "red" else "red"
        self.button.setText( self.color_text )    


if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    win = Window()
    win.show()
    qapp.exec_()