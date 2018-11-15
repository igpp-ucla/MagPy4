
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from mth import Mth
import math
from MagPy4UI import MatrixWidget
import functools

class WaveAnalysisUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Wave Analysis')
        Frame.resize(700,500)  

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
        self.matLayout.addWidget(ipFrame, 0, 1, 1, 1)
        self.matLayout.addWidget(trpFrame, 1, 0, 1, 1)
        self.matLayout.addWidget(tipFrame, 1, 1, 1, 1)
        self.matLayout.addWidget(trFrame, 2, 0, 1, 1)
        self.matLayout.addWidget(tiFrame, 2, 1, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.matLayout.addItem(spacer, 0, 100, 1, 1)

        self.layout.addLayout(self.matLayout)

        freqGroupLayout = QtWidgets.QHBoxLayout()

        freqFrame = QtWidgets.QGroupBox('Frequency Selection')
        freqLayout = QtWidgets.QGridLayout(freqFrame)

        #freqLayoutH = QtWidgets.QHBoxLayout(freqFrame)
        #freqLayout = QtWidgets.QVBoxLayout()
        #freqLayoutH.addLayout(freqLayout)
        #freqLayoutH.addStretch()

        self.minFreqLabel = QtWidgets.QLabel()
        self.maxFreqLabel = QtWidgets.QLabel()
        self.minFreqIndex = QtWidgets.QSpinBox()
        self.maxFreqIndex = QtWidgets.QSpinBox()
        self.updateButton = QtWidgets.QPushButton('Update')

        freqLayout.addWidget(self.minFreqIndex, 0, 0, 1, 1)
        freqLayout.addWidget(self.minFreqLabel, 0, 1, 1, 1)
        freqLayout.addWidget(self.maxFreqIndex, 1, 0, 1, 1)
        freqLayout.addWidget(self.maxFreqLabel, 1, 1, 1, 1)
        freqLayout.addWidget(self.updateButton, 2, 0, 1, 1)

        freqGroupLayout.addWidget(freqFrame)

        bornWolfFrame = QtWidgets.QGroupBox('Wave Analysis')
        bornWolfGrid = QtWidgets.QGridLayout(bornWolfFrame)

        self.ppLabel = QtWidgets.QLabel()
        self.ppmLabel = QtWidgets.QLabel()
        self.elipLabel = QtWidgets.QLabel()
        self.elipmLabel = QtWidgets.QLabel()
        self.azimLabel = QtWidgets.QLabel()

        bornWolfGrid.addWidget(QtWidgets.QLabel('Born-Wolf'), 0, 1, 1, 1)
        bornWolfGrid.addWidget(QtWidgets.QLabel('Joe Means'), 0, 2, 1, 1)
        bornWolfGrid.addWidget(QtWidgets.QLabel('% Polarization:'), 1, 0, 1, 1)
        bornWolfGrid.addWidget(self.ppLabel, 1, 1, 1, 1)
        bornWolfGrid.addWidget(self.ppmLabel, 1, 2, 1, 1)
        bornWolfGrid.addWidget(QtWidgets.QLabel('Ellipticity:'), 2, 0, 1, 1)
        bornWolfGrid.addWidget(self.elipLabel, 2, 1, 1, 1)
        bornWolfGrid.addWidget(self.elipmLabel, 2, 2, 1, 1)
        bornWolfGrid.addWidget(QtWidgets.QLabel('Azimuth Angle:'), 3, 0, 1, 1)
        bornWolfGrid.addWidget(self.azimLabel, 3, 1, 1, 1)

        #pp, ppm, elip, elipm, azim
        freqGroupLayout.addWidget(bornWolfFrame)

        #self.wolfText = QtGui.QTextBrowser()
        #freqGroupLayout.addWidget(self.wolfText)

        freqGroupLayout.addStretch()

        self.layout.addLayout(freqGroupLayout)

        self.layout.addStretch()


        botLayout = QtWidgets.QHBoxLayout()
        logButton = QtWidgets.QPushButton('Export Log')
        botLayout.addWidget(logButton)
        botLayout.addStretch()
        self.layout.addLayout(botLayout)

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

        self.ui.updateButton.clicked.connect(self.updateCalculations)
        self.ui.minFreqIndex.valueChanged.connect(functools.partial(self.updateLabel, self.ui.minFreqLabel))
        self.ui.maxFreqIndex.valueChanged.connect(functools.partial(self.updateLabel, self.ui.maxFreqLabel))

        # ya make freq sliders, half number of frequencys per number of bands selected, line gets plotted at actually frequency value (band is just index into fft array)
        # then hook them up to lines like u do for spectra selection
        # then let them select on the graph itself?? kinda annoying with multiple spectra plots, maybe do that later
        # just do another bisect search thing

        #make export to log just print everything out nicely formatted........ would be nice if u could just copy and paste what u want but ehhh
        # kinda annoying to use those qtextbrowser things. copy and paste doesnt work how u think it would inherently

        freqs = self.getDefaultFreqs()
        #m = len(ffts[0]) // 2
        m = len(freqs) // 2
        self.ui.minFreqIndex.setMinimum(0)
        self.ui.maxFreqIndex.setMinimum(0)
        self.ui.minFreqIndex.setMaximum(m)
        self.ui.maxFreqIndex.setMaximum(m)
        self.ui.minFreqIndex.setValue(0)
        self.ui.maxFreqIndex.setValue(m)

        self.ui.minFreqIndex.valueChanged.emit(0)#otherwise wont refresh first time

        self.updateCalculations() # should add update button later

    def updateLabel(self, label, val):
        freqs = self.getDefaultFreqs()
        label.setText(Mth.formatNumber(freqs[val]))

    def getDefaultFreqs(self):
        return self.spectra.getFreqs(self.ui.axesDropdowns[0].currentText(), 0)

    def updateCalculations(self):
        """ update all wave analysis values and corresponding UI elements """

        dstrs = [dd.currentText() for dd in self.ui.axesDropdowns]

        ffts = [self.spectra.getfft(dstr,0) for dstr in dstrs]

        fO = self.ui.minFreqIndex.value()
        fE = self.ui.maxFreqIndex.value()
        # ensure first frequency index is less than last and not the same value
        if fE < fO:
            fO,fE = fE,fO
        if abs(fO-fE) < 2:
            fE = fO + 2
        self.ui.minFreqIndex.setValue(fO)
        self.ui.maxFreqIndex.setValue(fE)
        
        #print(f'{fO} {fE} {ffts[0][fO]} {ffts[0][fE]}')
        #print(f'{fO} {fE} {freqs[fO]} {freqs[fE]}')

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
        trp = Mth.arpat(amat, realPower) # transformed real power
        trp = Mth.flip(trp)
        tip = Mth.arpat(amat, imagPower) # transformed imaginary power
        tip = Mth.flip(tip)
        trm = Mth.arpat(bmat, realPower) # transformed real matrix
        tim = Mth.arpat(bmat, imagPower) # transformed imaginary matrix

        self.ui.trpMat.setMatrix(trp)
        self.ui.tipMat.setMatrix(tip)
        self.ui.trMat.setMatrix(trm)
        self.ui.tiMat.setMatrix(tim)

        pp, ppm, elip, elipm, azim = self.bornWolf(trp,tip,trm,tim)

        self.ui.ppLabel.setText(Mth.formatNumber(pp))
        self.ui.ppmLabel.setText(Mth.formatNumber(ppm))
        self.ui.elipLabel.setText(Mth.formatNumber(elip))
        self.ui.elipmLabel.setText(Mth.formatNumber(elipm))
        self.ui.azimLabel.setText(Mth.formatNumber(azim))

        #self.updateBornAnalysis(pp, ppm, elip, elipm, azim)

    # this was directly imported from original magpy
    def bornWolf(self, trp, tip, trm, tim):
        """
		Given transformed versions of real and imaginary powers and matrices
		calculate polarization and ellipticity (both with joe means versions), and azimuth angle
		"""
        trj = trp[0][0] + trp[1][1]
        detj = trp[0][0] * trp[1][1] - trp[1][0] * trp[1][0] - tip[1][0] * tip[1][0]
        fnum = 1 - (4 * detj) / (trj * trj)
        if fnum <= 0:
            pp = 0
        else:
            pp = 100 * math.sqrt(fnum)
        vetm = trj * trj - 4.0 * detj
        eden = 1 if vetm < 0 else math.sqrt(vetm)
        fnum = 2 * tip[0][1] / eden
        if (trp[0][1] < -1e-10):
            elip = -1.0*math.tan(0.5*math.asin(fnum))
        else:
            elip = math.tan(0.5*math.asin(fnum))
        trj = trm[0][0] + trm[1][1]
        detj=trm[0][0]*trm[1][1]-trm[0][1]*trm[1][0]-tim[1][0]*tim[1][0]
        difm = trm[0][0] - trm[1][1]
        fnum = 1 - (4 * detj) / (trj * trj)
        ppm = 100 * math.sqrt(fnum)
        fnum = 2.0 * tim[0][1] / eden
        if fnum <= 0:
            fnum = 0
            pp = 0
        elipm = math.tan(0.5 * math.asin(fnum))
        fnum = 2.0 * trm[0][1]
        angle = fnum / difm
        azim = 0.5 * math.atan(angle) * Mth.R2D
        return pp, ppm, elip, elipm, azim

    # old magpy table display, slightly updated it but not using currently
    def updateBornAnalysis(self, pp, ppm, elip, elipm, azim):
        head = "<HTML><Table width='100%'><tr><th><td>Born-Wolf</td><td>Joe Means</td></th></tr>"
        polar = f"<tr><td>% Polarization: </td><td>{pp:+5.3f} </td><td> {ppm:5.3f}</td></tr>"
        ellip = f"<tr><td>Ellipticity(+RH,-LH): </td><td>{elip:+5.3f} </td><td> {elipm:5.3f}</td></tr>"
        angle = f"<tr><td>Azimuth Angle: </td><td>{azim:+5.3f} </td><td>  </td></tr>"
        html =  f"{head}{polar}{ellip}{angle}</table></HTML>"
        self.ui.wolfText.setText(html)

