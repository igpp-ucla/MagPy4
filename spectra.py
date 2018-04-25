

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(250,200)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        
        layout.addWidget(self.glw)

        # bandwidth label and spinbox
        bandWidthLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtGui.QLabel("Average Bandwidth")
        self.bandWidthSpinBox = QtGui.QSpinBox()
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setProperty("value", 3)
        bandWidthLayout.addWidget(bandWidthLabel)
        bandWidthLayout.addWidget(self.bandWidthSpinBox)
        bandWidthLayout.addStretch()

        self.processButton = QtWidgets.QPushButton('Process')
        self.processButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        layout.addLayout(bandWidthLayout)
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


        self.N = self.window.numpoints # todo make this be whatever size of selection is
        freq = self.calculateFreqList()

        self.ui.glw.clear()
        for magData in magDatas:
            #magData = magData.astype('>f8')
            fft = fftpack.rfft(magData.tolist())
            power = self.calculatePower(fft)

            pi = pg.PlotItem()
            pi.setLogMode(True, True) #todo redo ticks to be of format 10^x
            print(f'freqLen {len(freq)} powLen {len(power)}')

            pi.plot(freq, power[:len(freq)], pen=self.window.pens[0])

            self.ui.glw.addItem(pi)

            break

    def calculateFreqList(self):
        bw = self.ui.bandWidthSpinBox.value()
        km = int((bw + 1.0) * 0.5)
        nband = (self.N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - half + 1) #try to match power length
        C = self.N * self.window.resolution
        freq = np.arange(km, nfreq) / C
        #return np.log10(freq)
        if len(freq) < 2:
            print('Proposed spectra plot invalid!\nFrequency list has lass than 2 values')
            return None
        return freq

    def calculatePower(self, fft):
        bw = self.ui.bandWidthSpinBox.value()
        kmo = int((bw + 1) * 0.5)
        nband = (self.N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - bw + 1)

        C = 2 * self.window.resolution / self.N
        fsqr = [ft * ft for ft in fft]
        power = [0] * nfreq
        for n in range(nfreq):
            km = kmo + n
            kO = int(km - half)
            kE = int(km + half) + 1
            p = [C * (fsqr[2 * k - 1] + fsqr[2 * k]) for k in range(kO,kE)]
            power[n] = np.sum(p) / bw

        return power

    def calculateCoherence(self):
        pass


        


