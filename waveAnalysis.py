
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from mth import Mth
import math
from MagPy4UI import MatrixWidget

class WaveAnalysisUI(object):
        def setupUI(self, Frame, window):
            Frame.setWindowTitle('Wave Analysis')
            Frame.resize(1000,700)  

            self.hlayout = QtWidgets.QHBoxLayout(Frame)
            self.layout = QtWidgets.QVBoxLayout()

            self.axLayout = QtWidgets.QGridLayout()
            self.window = window
            defaultPlots = self.window.getDefaultPlotInfo()[0]
            axes = ['X','Y','Z','T']
            self.axesDropdowns = []
            for i,ax in enumerate(axes):
                dd = QtGui.QComboBox()
                self.axLayout.addWidget(QtWidgets.QLabel(ax),0,i,1,1)
                for s in self.window.DATASTRINGS:
                    if ax.lower() in s.lower():
                        dd.addItem(s)
                dd.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                self.axesDropdowns.append(dd)
                self.axLayout.addWidget(dd,1,i,1,1)

            spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.axLayout.addItem(spacer, 0, 100, 1, 1)

            self.layout.addLayout(self.axLayout)
            
            rpFrame = QtWidgets.QGroupBox('Real Power')
            rpLayout = QtWidgets.QVBoxLayout(rpFrame)
            self.rpMatrix = MatrixWidget()
            rpLayout.addWidget(self.rpMatrix)
            self.layout.addWidget(rpFrame)

            ipFrame = QtWidgets.QGroupBox('Imaginary Power')
            ipLayout = QtWidgets.QVBoxLayout(ipFrame)
            self.ipMatrix = MatrixWidget()
            ipLayout.addWidget(self.ipMatrix)
            self.layout.addWidget(ipFrame)


            self.layout.addStretch()

            self.hlayout.addLayout(self.layout)
            self.hlayout.addStretch()


class WaveAnalysis(QtWidgets.QFrame, WaveAnalysisUI):
    def __init__(self, spectra, window, parent=None):
        super(WaveAnalysis, self).__init__(parent)

        self.spectra = spectra
        self.window = window
        self.ui = WaveAnalysisUI()
        self.ui.setupUI(self, window)

        self.updateCalculations() # should add update button later


    def updateCalculations(self):
        dstrs = [dd.currentText() for dd in self.ui.axesDropdowns]
        #print(dstrs)

        ffts = [self.spectra.getfft(dstr) for dstr in dstrs]
        print(len(ffts[0]))

        # needs start and end frequencies sliders prob (for now can just use whole spectra)
        # need to correct this. start and end points aren't exactly great
        fO = 0
        fE = len(ffts[0]) // 2
        k = fE - fO - 1
        counts = np.array(range(int(k))) + 1
        steps = 2 * counts - 1
        #print(steps)

        ffts = [fft[fO:fE * 2] for fft in ffts]
        sqrs = [fft * fft for fft in ffts]

        deltaf = 2.0 / (self.spectra.maxN * self.spectra.maxN)

        ps = [sum([sq[i] + sq[i + 1] for i in steps]) * deltaf for sq in sqrs]
        #x,y  x,z  y,z
        axisPairs = [(ffts[0],ffts[1]),(ffts[0],ffts[2]),(ffts[1],ffts[2])]

        cs = [sum([fft0[i] * fft1[i] + fft0[i + 1] * fft1[i + 1] for i in steps]) * deltaf for fft0,fft1 in axisPairs]
        qs = [sum([fft0[i] * fft1[i + 1] - fft0[i + 1] * fft1[i] for i in steps]) * deltaf for fft0,fft1 in axisPairs]

        realPower = [[ps[0], cs[0], cs[1]], [cs[0], ps[1], cs[2]], [cs[1], cs[2], ps[2]]]
        imagPower = [[0.0, -qs[0], -qs[1]], [qs[0], 0.0, -qs[2]], [qs[1], qs[2], 0.0]]

        self.ui.rpMatrix.setMatrix(realPower)
        self.ui.ipMatrix.setMatrix(imagPower)

        #powSpectra = ps[0] + ps[1] + ps[2]
        #traAmp = sqrt(powSpectra)
        #comPow = ps[3]
        #comAmb = sqrt[ps[3]]
        #comRat = ps[3] / powSpectra

        avg = [np.mean(self.window.getData(dstr)) for dstr in dstrs]

        qqq = np.linalg.norm(qs)
        qqqd = avg[3]
        qqqp = np.linalg.norm(avg[:-1])
        qkem = np.array([qs[2] / qqq, -qs[1] / qqq, qs[0] / qqq]) # propogation direction
        qqqn = np.dot(qkem, avg[:-1])
        if qqqn < 0:
            qkem = qkem * -1
            qqqn = np.dot(qkem, avg[:-1])
        qqq = qqqn / qqqp
        qtem = Mth.R2D * math.acos(qqq) # field angle
        qqq = np.linalg.norm(cs)
        qdlm = np.array(cs[::-1] / qqq)
        qqqn = np.dot(qdlm, avg[:-1])
        qqq = qqqn / qqqp
        qalm = Mth.R2D * math.acos(qqq)


    def flip(a):   # flip and twist
        b = [[a[2][2], a[2][1], a[2][0]], [a[1][2], a[1][1], a[1][0]], [a[0][2], a[0][1], a[0][0]]]
        return b

    def arpat(a, b):
        # A * B * a^T
    #   at = numpy.transpose(a)
    #   c = b * a * at
        temp = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                temp[j][i] = 0
                for k in range(3):
                    temp[j][i] = b[k][i] * a[k][j] + temp[j][i]
        for i in range(3):
            for j in range(3):
                c[j][i] = 0
                for k in range(3):
                    c[j][i] = a[k][i] * temp[j][k] + c[j][i]
        return c
