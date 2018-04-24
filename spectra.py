

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(250,200)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        
        layout.addWidget(self.glw)

        self.processButton = QtWidgets.QPushButton('Process')

        layout.addWidget(self.processButton)


class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)

        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self)
        
        self.ui.processButton.clicked.connect(self.processData)

    def processData(self):
        keywords = ['BX','BY','BZ','BT']
        magDatas = []
        for kw in keywords:
            for dstr in self.window.DATASTRINGS:
                if kw.lower() in dstr.lower():
                    magDatas.append(self.window.DATADICT[dstr])
                    #print(f'found {dstr}')

        ffts = []
        for magData in magDatas:
            fft = fftpack.rfft(magData)
            print(fft)

        #pi = pg.PlotItem()
        #pi.plot(

        #self.ui.glw.addItem(pi)


        


