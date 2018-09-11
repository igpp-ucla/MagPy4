
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

        self.layout = QtWidgets.QVBoxLayout(Frame)

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
            
        self.matLayout = QtWidgets.QGridLayout()

        #self.rpMatrix = self.addMatrixBox('Real Power', self.matLayout, 0, 0, 1, 1)
        #self.ipMatrix = self.addMatrixBox('Imaginary Power', self.matLayout, 1, 0, 1, 1)

        self.rpMat, rpFrame = self.addMatrixBox('Real Power')
        self.ipMat, ipFrame = self.addMatrixBox('Imaginary Power')
        self.trpMat, trpFrame = self.addMatrixBox('Transformed Real Power')
        self.tipMat, tipFrame = self.addMatrixBox('Transformed Imaginary Power')
        self.trMat, trFrame = self.addMatrixBox('Transformed Real Matrix')
        self.tiMat, tiFrame = self.addMatrixBox('Transformed Imaginary Matrix')

        self.matLayout.addWidget(rpFrame, 0, 0, 1, 1)
        self.matLayout.addWidget(ipFrame, 1, 0, 1, 1)
        self.matLayout.addWidget(trpFrame, 2, 0, 1, 1)
        self.matLayout.addWidget(tipFrame, 2, 1, 1, 1)
        self.matLayout.addWidget(trFrame, 3, 0, 1, 1)
        self.matLayout.addWidget(tiFrame, 3, 1, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.matLayout.addItem(spacer, 0, 100, 1, 1)

        self.layout.addLayout(self.matLayout)
        self.layout.addStretch()

    def addMatrixBox(self, name):
        frame = QtWidgets.QGroupBox(name)
        layout = QtWidgets.QVBoxLayout(frame)
        mat = MatrixWidget()
        layout.addWidget(mat)
        return mat, frame

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
        #print(len(ffts[0]))

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

        self.ui.rpMat.setMatrix(realPower)
        self.ui.ipMat.setMatrix(imagPower)

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

        #means transformation matrix
        yx = qkem[1] * avg[2] - qkem[2] * avg[1]
        yy = qkem[2] * avg[0] - qkem[0] * avg[2]
        yz = qkem[0] * avg[1] - qkem[1] * avg[0]
        qyxyz = np.linalg.norm([yx, yy, yz])
        yx = yx / qyxyz
        yy = yy / qyxyz
        yz = yz / qyxyz
        xx = yy * qkem[2] - yz * qkem[1]
        xy = yz * qkem[0] - yx * qkem[2]
        xz = yx * qkem[1] - yy * qkem[0]
        bmat = [[xx, yx, qkem[0]], [xy, yy, qkem[1]], [xz, yz, qkem[2]]]
        duhh, amat = np.linalg.eigh(np.transpose(realPower), UPLO="U")
        #self.thbk, self.thkk = getAngles(amat, avg, qkem)

        # Transformed Values Spectral Matrices 
        trp = Mth.arpat(amat, realPower)
        trp = Mth.flip(trp)
        tip = Mth.arpat(amat, imagPower)
        tip = Mth.flip(tip)
        trm = Mth.arpat(bmat, realPower)
        tim = Mth.arpat(bmat, imagPower)

        self.ui.trpMat.setMatrix(trp)
        self.ui.tipMat.setMatrix(tip)
        self.ui.trMat.setMatrix(trm)
        self.ui.tiMat.setMatrix(tim)

        # born-wolf analysis
        #self.pp, self.ppm, self.elip, self.elipm, self.azim = bornWolf(rpp, ipp, rpmp, ipmp)

