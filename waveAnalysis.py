
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from mth import Mth
import math
from MagPy4UI import MatrixWidget
import functools
import os

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
        self.logButton = QtWidgets.QPushButton('Export Log')
        botLayout.addWidget(self.logButton)
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
        self.ui.logButton.clicked.connect(self.exportLog)

        freqs = self.getDefaultFreqs()
        m = len(freqs)
        self.ui.minFreqIndex.setMinimum(0)
        self.ui.maxFreqIndex.setMinimum(0)
        self.ui.minFreqIndex.setMaximum(m-1)
        self.ui.maxFreqIndex.setMaximum(m-1)
        self.ui.minFreqIndex.setValue(0)
        self.ui.maxFreqIndex.setValue(m-1)

        self.ui.minFreqIndex.valueChanged.emit(0)#otherwise wont refresh first time

        self.updateCalculations() # should add update button later

    def getTableData(self):
        # Create a dictionary of matrix titles and their arrays
        matrixBoxes = [self.ui.rpMat, self.ui.ipMat, self.ui.trpMat, self.ui.tipMat, 
            self.ui.trMat, self.ui.tiMat]
        matrixNames = ['Real Power', 'Imaginary Power', 'Transformed Real Power',
            'Transformed Imaginary Power', 'Transformed Real Matrix',
            'Transformed Imaginary Matrix']
        matrices = {}
        for i in range(0, len(matrixBoxes)):
            matrices[matrixNames[i]] = matrixBoxes[i].getMatrix()

        # Recreate Born-Wolf and Joe Means results table in an array
        resultsTable = [['', 'Born-Wolf', 'Joe Means'],
            ['% Polarization:', self.ui.ppLabel.text(), self.ui.ppmLabel.text()],
            ['Ellipticity:', self.ui.elipLabel.text(), self.ui.elipmLabel.text()], 
            ['Azimuth Angle:', self.ui.azimLabel.text(), '']]

        # Get and format info about file, time range, and wave analysis parameters
        fileName = self.window.getFileNameString()

        timeFmtStr = 'yyyy MMM dd HH:mm:ss.zzz'
        startTime = self.spectra.ui.timeEdit.start.dateTime().toString(timeFmtStr)
        endTime = self.spectra.ui.timeEdit.end.dateTime().toString(timeFmtStr)
        timeRangeStr = 'Time Range: ' + startTime + ' to ' + endTime + '\n'

        startFreq = self.ui.minFreqLabel.text()
        endFreq = self.ui.maxFreqLabel.text()
        freqRangeStr = 'Min freq: ' + startFreq + '\nMax freq: ' + endFreq

        axesStr = 'Axes: '
        for a in self.ui.axesDropdowns:
            axesStr += a.currentText() + ', '
        axesStr = axesStr[:-2] # Remove extra comma at end

        logInfo = ['File(s): '+fileName, timeRangeStr, axesStr, freqRangeStr]

        return logInfo, matrices, resultsTable

    def writeMatrix(self, f, M):
        # Write matrix to file with only one row per line and padded entries
        for row in M:
            for entry in row:
                f.write(str(entry).ljust(15))
                f.write(' ')
            f.write('\n')
        f.write('\n')

    def exportLog(self):
        logInfo, matrices, results = self.getTableData()

        # Save file dialog
        defaultSfx = '.txt'
        QQ = QtGui.QFileDialog(self)
        QQ.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        path = os.path.expanduser(".")
        QQ.setDirectory(path)
        fullname = QQ.getSaveFileName(parent=None, directory=path, caption="Save Data", filter='TXT file (*.txt)')
        if fullname is None:
            print('Save failed')
            return
        if fullname[0] == '':
            print('Save cancelled')
            return

        # If file name doesn't end with default suffix, add it before saving
        filename = fullname[0]
        if filename.endswith(defaultSfx) == False:
            filename += defaultSfx

        # Write logInfo, each matrix, and results to file
        f = open(filename, 'w')
        for entry in logInfo:
            f.write(entry)
            f.write('\n')
        f.write('\n')
        for k in list(matrices.keys()):
            f.write(k + '\n')
            self.writeMatrix(f, matrices[k])
        self.writeMatrix(f, results)
        f.close()

    def updateLabel(self, label, val):
        freqs = self.getDefaultFreqs()
        label.setText(Mth.formatNumber(freqs[val]))

    def getDefaultFreqs(self):
        return self.spectra.getFreqs(self.ui.axesDropdowns[0].currentText(), 0)

    def updateCalculations(self):
        """ update all wave analysis values and corresponding UI elements """

        en = self.window.currentEdit
        dstrs = [dd.currentText() for dd in self.ui.axesDropdowns]
        ffts = [self.spectra.getfft(dstr,en) for dstr in dstrs]

        fO = self.ui.minFreqIndex.value()
        fE = self.ui.maxFreqIndex.value()

        # ensure first frequency index is less than last and not the same value
        if fE < fO:
            fO,fE = fE,fO
        if abs(fO-fE) < 2:
            fE = fO + 2
        self.ui.minFreqIndex.setValue(fO)
        self.ui.maxFreqIndex.setValue(fE)

        # Generate the indices (relative to fO*2) in ffts arrays corresponding to a freq num
        # since each freq corresponds to a (cos, sin) pair in the fast fourier transformed values
        # Starts at index 1 (see scipy rfft documentation)
        k = fE - fO - 1
        counts = np.array(range(int(k))) + 1 
        steps = 2 * counts - 1

        ffts = [fft[fO*2:fE*2] for fft in ffts]
        deltaf = 2.0 / (self.spectra.maxN * self.spectra.maxN)

        # Compute real part of matrix's diagonals
        sqrs = [fft * fft for fft in ffts]
        ps = [sum([sq[i] + sq[i + 1] for i in steps]) * deltaf for sq in sqrs]

        # (x,y), (x,z), (y,z)
        axisPairs = [(ffts[0],ffts[1]),(ffts[0],ffts[2]),(ffts[1],ffts[2])]
        # Compute real part of matrix's off-diagonals and the elements in the
        # imaginary matrix that can be used to compute the wave normal vector
        cs = [sum([fft0[i] * fft1[i] + fft0[i + 1] * fft1[i + 1] for i in steps]) * deltaf for fft0,fft1 in axisPairs]
        qs = [sum([fft0[i] * fft1[i + 1] - fft0[i + 1] * fft1[i] for i in steps]) * deltaf for fft0,fft1 in axisPairs]

        realPower = [[ps[0], cs[0], cs[1]], [cs[0], ps[1], cs[2]], [cs[1], cs[2], ps[2]]]
        imagPower = [[0.0, -qs[0], -qs[1]], [qs[0], 0.0, -qs[2]], [qs[1], qs[2], 0.0]]

        self.ui.rpMat.setMatrix(realPower)
        self.ui.ipMat.setMatrix(imagPower)

        # Compute the average field for each dstr within the given time range
        avg = []
        for dstr in dstrs:
            sI, eI = self.spectra.getIndices(dstr, en)
            avg.append(np.mean(self.window.getData(dstr)[sI:eI]))

        # Wave propogation direction
        qqq = np.linalg.norm(qs)
        qkem = np.array([-qs[2] / qqq, qs[1] / qqq, -qs[0] / qqq])

        qqqd = avg[3]
        qqqp = np.linalg.norm(avg[:-1])
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

        # Means transformation matrix
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
        duhh, amat = np.linalg.eigh(realPower, UPLO="U")
        bmat = np.transpose(bmat)
        amat = np.transpose(amat)

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
        eden = 1 if vetm <= 0 else math.sqrt(vetm)
        fnum = 2 * tip[0][1] / eden
        if (trp[0][1] < 0):
            elip = -1.0*math.tan(0.5*math.asin(fnum))
        else:
            elip = math.tan(0.5*math.asin(fnum))
        trj = trm[0][0] + trm[1][1]
        detj=trm[0][0]*trm[1][1]-trm[0][1]*trm[1][0]-tim[1][0]*tim[1][0]
        difm = trm[0][0] - trm[1][1]
        fnum = 1 - (4 * detj) / (trj * trj)
        ppm = 100 * math.sqrt(fnum)
        fnum = 2.0 * tim[0][1] / eden
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

