
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from dynamicSpectra import DynamicAnalysisTool, SpectrogramPlotItem, PhaseSpectrogram
from layoutTools import BaseLayout
from pyqtgraphExtensions import StackedAxisLabel
from scipy import fftpack
import pyqtgraph as pg

import numpy as np
from mth import Mth
import bisect
import math
from MagPy4UI import MatrixWidget, NumLabel
import functools
import os

class WaveAnalysisUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Wave Analysis')
        Frame.resize(700,500)  

        self.layout = QtWidgets.QGridLayout(Frame)
        prncpFrame = QtWidgets.QGroupBox('Principal Axis Analysis')
        self.prncplLayout = QtWidgets.QVBoxLayout(prncpFrame)

        self.axLayout = QtWidgets.QGridLayout()
        self.window = window
        defaultPlots = self.window.getDefaultPlotInfo()[0]
        axes = ['X','Y','Z']
        self.axesDropdowns = []
        for i,ax in enumerate(axes):
            dd = QtGui.QComboBox()
            dlbl = QtWidgets.QLabel(ax)
            self.axLayout.addWidget(dlbl,0,i,1,1)
            # Add elements into comboboxes
            addedItems = []
            for s in self.window.DATASTRINGS:
                if ax.lower() in s.lower():
                    dd.addItem(s)
                    addedItems.append(s)
            if addedItems == []: # If no valid combination, add all and set
                dd.addItems(self.window.DATASTRINGS)
                dd.setCurrentText(self.window.DATASTRINGS[i])
            dd.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            dlbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

            self.axesDropdowns.append(dd)
            self.axLayout.addWidget(dd,1,i,1,1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.axLayout.addItem(spacer, 0, 150, 1, 1)

        self.layout.addLayout(self.axLayout, 0, 0, 1, 1)
            
        self.matLayout = QtWidgets.QGridLayout()

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

        self.layout.addLayout(self.matLayout, 1, 0, 1, 1)

        freqGroupLayout = QtWidgets.QHBoxLayout()

        freqFrame = QtWidgets.QGroupBox('Frequency Selection')
        freqLayout = QtWidgets.QGridLayout(freqFrame)

        self.minFreqLabel = QtWidgets.QLabel()
        self.maxFreqLabel = QtWidgets.QLabel()
        self.minFreqIndex = QtWidgets.QSpinBox()
        self.maxFreqIndex = QtWidgets.QSpinBox()
        self.maxFreqIndex.setFixedWidth(100)
        self.minFreqIndex.setFixedWidth(100)
        self.updateButton = QtWidgets.QPushButton('Update')

        freqLayout.addWidget(self.minFreqIndex, 0, 0, 1, 1)
        freqLayout.addWidget(self.minFreqLabel, 1, 0, 1, 1)
        freqLayout.addWidget(self.maxFreqIndex, 0, 1, 1, 1)
        freqLayout.addWidget(self.maxFreqLabel, 1, 1, 1, 1)
        freqLayout.addWidget(self.updateButton, 0, 3, 1, 1)
        freqFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        freqGroupLayout.addWidget(freqFrame)

        # Setup Born-Wolf Analysis grid
        bornWolfFrame = QtWidgets.QGroupBox('Born-Wolf Analysis')
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

        # Setup Joe Means Analysis table
        joeMeansFrame = QtWidgets.QGroupBox('Joe Means Analysis')
        joeMeansGrid = QtWidgets.QGridLayout(joeMeansFrame)

        self.prop = QtWidgets.QLabel()
        self.jmAngle = QtWidgets.QLabel()
        self.linVar = QtWidgets.QLabel()
        self.lvAngle = QtWidgets.QLabel()

        joeMeansGrid.addWidget(QtWidgets.QLabel('Propagation: '), 0, 0, 1, 1)
        joeMeansGrid.addWidget(self.prop, 0, 1, 1, 1)
        joeMeansGrid.addWidget(QtWidgets.QLabel('Angle: '), 0, 2, 1, 1)
        joeMeansGrid.addWidget(self.jmAngle, 0, 3, 1, 1)
        joeMeansGrid.addWidget(QtWidgets.QLabel('Linear Var: '), 1, 0, 1, 1)
        joeMeansGrid.addWidget(self.linVar, 1, 1, 1, 1)
        joeMeansGrid.addWidget(QtWidgets.QLabel('Angle: '), 1, 2, 1, 1)
        joeMeansGrid.addWidget(self.lvAngle, 1, 3, 1, 1)

        # Set up eigenvectors/values table
        eigenLayout = QtWidgets.QHBoxLayout()
        self.eigenVecs, eigenVecFrame = self.addMatrixBox('Eigenvectors')
        eigenValsFrame = QtWidgets.QGroupBox('Eigenvalues')
        eigenValsGrid = QtWidgets.QVBoxLayout(eigenValsFrame)

        eigenValsGrid.setAlignment(QtCore.Qt.AlignCenter)
        self.ev1, self.ev2, self.ev3 = QtWidgets.QLabel(), QtWidgets.QLabel(), QtWidgets.QLabel()
        for ev in [self.ev1, self.ev2, self.ev3]:
            eigenValsGrid.addWidget(ev)
            ev.setAlignment(QtCore.Qt.AlignCenter)

        eigenLayout.addStretch()
        eigenLayout.addWidget(eigenVecFrame)
        eigenLayout.addStretch()
        eigenLayout.addWidget(eigenValsFrame)
        eigenLayout.addStretch()

        # Setup principal axis analysis grid
        self.prncplLayout.addLayout(eigenLayout)
        self.prncplLayout.addWidget(joeMeansFrame)
        self.prncplLayout.addWidget(bornWolfFrame)
        self.layout.addWidget(prncpFrame, 1, 1, 1, 1)
        self.layout.addLayout(freqGroupLayout, 2, 0, 1, 2)

        # Setup export button in layout
        freqGroupLayout.addStretch()
        self.logButton = QtWidgets.QPushButton('Export Log')
        self.logButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.layout.addWidget(self.logButton, 3, 0, 1, 1)

        # Center titles above each principal axis analysis results group
        for grp in [prncpFrame, joeMeansFrame, bornWolfFrame, eigenVecFrame, eigenValsFrame]:
            grp.setAlignment(QtCore.Qt.AlignCenter)

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

        # Set up eigenvector/eigenvalue matrix objects
        eigenVecs = self.ui.eigenVecs.getMatrix()
        eigenVals = [[ev.text()] for ev in [self.ui.ev1, self.ui.ev2, self.ui.ev3]]

        # Recreate Joe Means results matrix in an array
        jmResults = [['Propagation: ', self.ui.prop.text(), 'Angle: ',
            self.ui.jmAngle.text()], ['Linear Var: ', self.ui.linVar.text(),
            'Angle: ', self.ui.lvAngle.text()]]

        # Recreate Born-Wolf and Joe Means results table in an array
        bwResults = [['', 'Born-Wolf', 'Joe Means'],
            ['% Polarization:', self.ui.ppLabel.text(), self.ui.ppmLabel.text()],
            ['Ellipticity:', self.ui.elipLabel.text(), self.ui.elipmLabel.text()], 
            ['Azimuth Angle:', self.ui.azimLabel.text(), '']]

        # Create tuple pairs of results matrices and titles in principal axis analysis
        resultsTable = [('Eigenvectors', eigenVecs), ('Eigenvalues', eigenVals),
            ('Joe Means analysis', jmResults), ('Born-Wolf analysis', bwResults)]

        # Get and format info about file, time range, and wave analysis parameters
        fileName = 'unknown'
        if len(self.window.FIDs) > 0:
            names = [os.path.split(FID.name)[1] for FID in self.window.FIDs]
            fileName = ', \n'.join(names)
        elif self.window.cdfName:
            fileName = self.window.cdfName

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

        filename = self.window.saveFileDialog()

        # Write logInfo, each matrix, and results to file
        f = open(filename, 'w')
        for entry in logInfo:
            f.write(entry)
            f.write('\n')
        f.write('\n')
        for k in list(matrices.keys()):
            f.write(k + '\n')
            self.writeMatrix(f, matrices[k])

        for lbl, mat in results:
            f.write(lbl + '\n')
            self.writeMatrix(f, mat)
        f.close()

    def vectorLabel(self, vec):
        txt = ''
        prec = 3
        for i in vec:
            txt += str(round(i, 3)) + '\t'
        return txt

    def updateLabel(self, label, val):
        freqs = self.getDefaultFreqs()
        label.setText(Mth.formatNumber(freqs[val]))

    def getDefaultFreqs(self):
        return self.spectra.getFreqs(self.ui.axesDropdowns[0].currentText(), 0)

    def updateCalculations(self):
        """ Update all wave analysis values and corresponding UI elements """

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

        # Each fft value corresponds to a cos,sin pair (real, imag), so
        # start at fO*2
        ffts = [fft[fO*2:fE*2+1] for fft in ffts]
        deltaf = 2.0 / (self.spectra.maxN * self.spectra.maxN)

        # Complex version of spectral matrix calculations
        cffts = [self.spectra.fftToComplex(fft) for fft in ffts]
        numIndices = len(cffts[0]) - 1

        # Compute H * H_star (freq * freq_conjugate)
        mats = self.spectra.computeSpectralMats(cffts)

        # Sum over each matrix
        sumMat = sum(mats) * deltaf

        # Cospectrum matrix is the real part of the matrix
        # Quadrature matrix is the imaginary part of the matrix
        realPower = sumMat.real
        imagPower = sumMat.imag
        # Extract relavent values from spectral matrix
        numPairs = [(0, 1), (0, 2), (1, 2)]
        qs = [imagPower[i][j] for i,j in numPairs]
        cs = [realPower[i][j] for i,j in numPairs]

        prec = 6
        self.ui.rpMat.setMatrix(np.round(realPower, prec))
        self.ui.ipMat.setMatrix(np.round(imagPower, prec))

        # Compute the average field for each dstr within the given time range
        avg = []
        for dstr in dstrs:
            sI, eI = self.spectra.getIndices(dstr, en)
            avg.append(np.mean(self.window.getData(dstr)[sI:eI]))

        # Wave propogation direction
        qqq = np.linalg.norm(qs)
        qkem = np.array([qs[2] / qqq, -qs[1] / qqq, qs[0] / qqq])

        qqqp = np.linalg.norm(avg)
        qqqn = np.dot(qkem, avg)
        if qqqn < 0:
            qkem = qkem * -1
            qqqn = np.dot(qkem, avg)
        qqq = qqqn / qqqp
        qtem = Mth.R2D * math.acos(qqq) # field angle
        qqq = np.linalg.norm(cs)
        qdlm = np.array(cs[::-1] / qqq)
        qqqn = np.dot(qdlm, avg)
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

        self.ui.trpMat.setMatrix(np.round(trp, prec))
        self.ui.tipMat.setMatrix(np.round(tip, prec))
        self.ui.trMat.setMatrix(np.round(trm, prec))
        self.ui.tiMat.setMatrix(np.round(tim, prec))

        pp, ppm, elip, elipm, azim = self.bornWolf(trp,tip,trm,tim)

        self.ui.ppLabel.setText(Mth.formatNumber(pp))
        self.ui.ppmLabel.setText(Mth.formatNumber(ppm))
        self.ui.elipLabel.setText(Mth.formatNumber(elip))
        self.ui.elipmLabel.setText(Mth.formatNumber(elipm))
        self.ui.azimLabel.setText(Mth.formatNumber(azim))

        self.ui.prop.setText(self.vectorLabel(qkem))
        self.ui.jmAngle.setText(str(round(qtem, 3)))
        self.ui.linVar.setText(self.vectorLabel(qdlm))
        self.ui.lvAngle.setText(str(round(qalm, 3)))

        # Set eigenvector, eigenvalue, and eigenvalue labels in correct order/arrngment
        amat = np.round(np.transpose(amat), prec)
        self.ui.eigenVecs.setMatrix(np.array([amat[:,2], amat[:,1], amat[:,0]]))
        for ev, lbl in zip(duhh[::-1], [self.ui.ev1, self.ui.ev2, self.ui.ev3]):
            lbl.setText(str(round(ev, 5)))

    def bornWolf(self, trp, tip, trm, tim):
        """
		Given transformed versions of real and imaginary powers and matrices
		calculate polarization and ellipticity (both with joe means versions), and azimuth angle
		"""
        # Calculate polarization/ellipticity parameters by Born-Wolf
        trj = trp[0][0] + trp[1][1]
        detj = trp[0][0] * trp[1][1] - trp[1][0] * trp[1][0] - tip[1][0] * tip[1][0]
        fnum = 1 - (4 * detj) / (trj * trj)
        if fnum <= 0:
            fnum = 0.0
            pp = 0.0
        else:
            pp = 100 * math.sqrt(fnum)
        vetm = trj * trj - 4.0 * detj
        eden = 1 if vetm <= 0 else math.sqrt(vetm)
        fnum = 2 * tip[0][1] / eden
        if (trp[0][1] < 0):
            elip = -1.0*math.tan(0.5*math.asin(fnum))
        else:
            elip = math.tan(0.5*math.asin(fnum))

        # Calculate polarization/ellipticity parameters by Joe Means method
        trj = trm[0][0] + trm[1][1]
        detj=trm[0][0]*trm[1][1]-trm[0][1]*trm[1][0]-tim[1][0]*tim[1][0]
        fnum = 1 - (4 * detj) / (trj * trj)
        if fnum <= 0:
            fnum = 0
            ppm = 0
        else:
            ppm = 100 * math.sqrt(fnum)
        vetm = trj * trj - 4.0 * detj
        eden = 1 if vetm <= 0 else math.sqrt(vetm)
        fnum = 2.0 * tim[0][1] / eden
        elipm = math.tan(0.5 * math.asin(fnum))

        # Calculate azimuth angle
        fnum = 2.0 * trm[0][1]
        difm = trm[0][0] - trm[1][1]
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

class DynamicWaveUI(BaseLayout):
    def setupUI(self, Frame, window, params):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Wave Analysis')
        Frame.resize(800, 775)
        layout = QtWidgets.QGridLayout(Frame)

        # Setup time edits / status bar and the graphics frame
        lt, self.timeEdit, self.statusBar = self.getTimeStatusBar()
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())
        self.glw = self.getGraphicsGrid(window)

        # Set up user-set parameters interface
        settingsLt = self.setupSettingsLt(Frame, window, params)

        layout.addLayout(settingsLt, 0, 0, 1, 1)
        layout.addWidget(self.gview, 1, 0, 1, 1)
        layout.addLayout(lt, 2, 0, 1, 1)

    def initVars(self, window):
        # Initialize the number of data points in selected range
        minTime, maxTime = window.getTimeTicksFromTimeEdit(self.timeEdit)
        times = window.getTimes(self.vectorBoxes[0].currentText(), 0)[0]
        startIndex = window.calcDataIndexByTime(times, minTime)
        endIndex = window.calcDataIndexByTime(times, maxTime)
        nPoints = abs(endIndex-startIndex)
        self.fftDataPts.setText(str(nPoints))

    def setupSettingsLt(self, Frame, window, params):
        layout = QtWidgets.QGridLayout()

        # Set up plot type parameters
        self.waveParam = QtWidgets.QComboBox()
        self.waveParam.addItems(params)
        self.waveParam.currentTextChanged.connect(self.plotTypeToggled)
        self.addPair(layout, 'Plot Type: ', self.waveParam, 0, 0, 1, 1)

        # Set up axis vector dropdowns
        vecLt = QtWidgets.QHBoxLayout()
        self.vectorBoxes = []
        layout.addWidget(QtWidgets.QLabel('Vector: '), 1, 0, 1, 1)
        for i in range(0, 3):
            box = QtWidgets.QComboBox()
            vecLt.addWidget(box)
            self.vectorBoxes.append(box)
        layout.addLayout(vecLt, 1, 1, 1, 1)

        # Get axis vector variables to add to boxes
        allDstrs = window.DATASTRINGS[:]
        axisDstrs = Frame.getAxesStrs(allDstrs)
        # If missing an axis variable, use all dstrs as default
        lstLens = list(map(len, axisDstrs))
        if min(lstLens) == 0:
            defaultDstrs = allDstrs[0:3]
            for dstr, box in zip(defaultDstrs, self.vectorBoxes):
                box.addItems(allDstrs)
                box.setCurrentText(dstr)
        else: # Otherwise use defaults
            for dstrLst, box in zip(axisDstrs, self.vectorBoxes):
                box.addItems(dstrLst)

        # Setup data points indicator
        self.fftDataPts = QtWidgets.QLabel()
        ptsTip = 'Total number of data points within selected time range'
        self.addPair(layout, 'Data Points: ', self.fftDataPts, 2, 0, 1, 1, ptsTip)

        # Set up FFT parameters layout
        fftLt = self.setupFFTLayout()
        layout.addLayout(fftLt, 0, 3, 3, 1)

        # Set up y-axis scale mode box above range layout
        scaleTip = 'Scaling mode that will be used for y-axis (frequencies)'
        self.scaleModeBox = QtWidgets.QComboBox()
        self.scaleModeBox.addItems(['Linear', 'Logarithmic'])
        self.addPair(layout, 'Scaling: ', self.scaleModeBox, 0, 5, 1, 1, scaleTip)

        # Set up toggle/boxes for setting color gradient ranges
        rangeLt = self.setRangeLt()
        layout.addLayout(rangeLt, 1, 5, 2, 2)

        # Add in spacers between columns
        for col in [2, 4]:
            for row in range(0, 3):
                spcr = self.getSpacer(5)
                layout.addItem(spcr, row, col, 1, 1)

        # Initialize default values
        self.valRngToggled(False)
        self.plotTypeToggled(self.waveParam.currentText())

        # Add in update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        self.updtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        layout.addWidget(self.updtBtn, 2, 7, 1, 1)

        return layout

    def setupFFTLayout(self):
        layout = QtWidgets.QGridLayout()
        row = 0
        # Set up fft parameter spinboxes
        self.fftShift = QtWidgets.QSpinBox()
        self.fftInt = QtWidgets.QSpinBox()
        self.fftInt.setMaximum(1e10)

        self.fftInt.setValue(256)
        self.fftShift.setValue(64)

        # Set up bandwidth spinbox
        self.bwBox = QtWidgets.QSpinBox()
        self.bwBox.setSingleStep(2)
        self.bwBox.setValue(3)
        self.bwBox.setMinimum(1)

        fftIntTip = 'Number of data points to use per FFT calculation'
        shiftTip = 'Number of data points to move forward after each FFT calculation'
        scaleTip = 'Scaling mode that will be used for y-axis (frequencies)'

        # Add fft settings boxes and labels to layout
        self.addPair(layout, 'FFT Interval: ', self.fftInt, row, 0, 1, 1, fftIntTip)
        self.addPair(layout, 'FFT Shift: ', self.fftShift,  row+1, 0, 1, 1, shiftTip)
        self.addPair(layout, 'Bandwidth: ', self.bwBox, row+2, 0, 1, 1, scaleTip)

        return layout

    def addTimeInfo(self, timeRng, window):
        # Convert time ticks to tick strings
        startTime, endTime = timeRng
        startStr = window.getTimestampFromTick(startTime)
        endStr = window.getTimestampFromTick(endTime)

        # Remove day of year
        startStr = startStr[:4] + startStr[8:]
        endStr = endStr[:4] + endStr[8:]

        # Create time label widget and add to grid layout
        timeLblStr = 'Time Range: ' + startStr + ' to ' + endStr
        self.timeLbl = pg.LabelItem(timeLblStr)
        self.timeLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.glw.nextRow()
        self.glw.addItem(self.timeLbl, col=0)

    def setRangeLt(self):
        self.selectToggle = QtWidgets.QCheckBox('Set Range:')
        self.selectToggle.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        rangeLt = QtWidgets.QGridLayout()

        rngTip = 'Toggle to set max/min values represented by color gradient'
        self.selectToggle.setToolTip(rngTip)

        minTip = 'Minimum value represented by color gradient'
        maxTip = 'Maximum value represented by color gradient'

        self.valueMin = QtWidgets.QDoubleSpinBox()
        self.valueMax = QtWidgets.QDoubleSpinBox()

        # Set spinbox defaults
        for box in [self.valueMax, self.valueMin]:
            box.setMinimum(-100)
            box.setMaximum(100)
            box.setFixedWidth(85)

        spc = '       ' # Spaces that keep spinbox lbls aligned w/ chkbx lbl

        rangeLt.addWidget(self.selectToggle, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.maxLbl = self.addPair(rangeLt, 'Max: ', self.valueMax, 0, 1, 1, 1, maxTip)
        self.minLbl = self.addPair(rangeLt, 'Min: ', self.valueMin, 1, 1, 1, 1, minTip)

        # Connects checkbox to func that enables/disables rangeLt's items
        self.selectToggle.toggled.connect(self.valRngToggled)
        return rangeLt

    def valRngToggled(self, val):
        self.valueMax.setEnabled(val)
        self.valueMin.setEnabled(val)
        self.minLbl.setEnabled(val)
        self.maxLbl.setEnabled(val)
    
    def plotTypeToggled(self, val):
        # Update value spinbox min/max values and deselect
        self.valRngToggled(False)
        self.selectToggle.setChecked(False)
        minVal, maxVal = (-1, 1)
        step = 0.1
        prefix = ''
        if 'Angle' in val:
            minVal, maxVal = (-180, 180)
            step = 10
        elif 'Power' in val:
            minVal, maxVal = (-100, 100)
            step = 1
            prefix = '10^'

        for box in [self.valueMax, self.valueMin]:
            box.setMinimum(minVal)
            box.setMaximum(maxVal)
            box.setSingleStep(step)
            box.setPrefix(prefix)

class DynamicWave(QtGui.QFrame, DynamicWaveUI, DynamicAnalysisTool):
    def __init__(self, window, parent=None):
        super(DynamicWave, self).__init__(parent)
        DynamicAnalysisTool.__init__(self)
        self.ui = DynamicWaveUI()
        self.window = window
        self.wasClosed = False

        # Default settings / labels for each plot type
        self.defParams = { # Value range, grad label, grad label units
            'Azimuth Angle' : ((-90, 90), 'Azimuth Angle', 'Degrees'),
            'Ellipticity (Means)' : ((-1.0, 1.0), 'Ellipticity', None),
            'Ellipticity (Born-Wolf)' : ((0, 1.0), 'Ellipticity', None),
            'Propagation Angle (Means)' : ((0, 90), 'Angle', 'Degrees'),
            'Propagation Angle (BK)' : ((0, 90), 'Angle', 'Degrees'),
            'Power Spectra Trace' : (None, 'Log Power', 'nT^2/Hz'),
            'Compressional Power' : (None, 'Log Power', 'nT^2/Hz')
        }

        self.titleDict = {}
        for key in self.defParams.keys():
            self.titleDict[key] = key
        self.titleDict['Power Spectra Trace'] = 'Trace Power Spectral Density'
        self.titleDict['Propagation Angle (BK)'] = 'Minimum Variance Angle'
        self.titleDict['Compressional Power'] = 'Compressional Power Spectral Density'

        # Sorts plot type names into groups
        self.plotGroups = {'Angle' : [], 'Ellipticity' : [], 'Power' : []}
        for plotType in self.defParams.keys():
            for kw in self.plotGroups.keys():
                if kw in plotType:
                    self.plotGroups[kw].append(plotType)
                    break

        self.currIndices = None # Indices to be used for current calculation
        self.prevIndices = None # Indices used for last calculation

        self.fftDict = {} # Stores dicts for each dstr, second key is indices
        self.avgDict = {}

        self.lastCalc = None # Stores last calculated times, freqs, values

        self.ui.setupUI(self, window, self.defParams.keys())
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

        self.preWindow = None # Window used to pre-select information
        self.showPreSelectWin()

    def closeEvent(self, ev):
        self.close()
        self.closePreSelectWin()
        self.window.endGeneralSelect()

    def setUserSelections(self):
        if self.preWindow:
            # Set UI's values to match those in the preselect window & close it
            plotType, vectorDstrs, scaling, bw = self.preWindow.getParams()
            self.ui.waveParam.setCurrentText(plotType)
            self.ui.scaleModeBox.setCurrentText(scaling)
            self.ui.bwBox.setValue(bw)
            for box, axStr in zip(self.ui.vectorBoxes, vectorDstrs):
                box.setCurrentText(axStr)
            self.closePreSelectWin()

    def update(self):
        fftInt = self.ui.fftInt.value()
        fftShift = self.ui.fftShift.value()
        plotType = self.ui.waveParam.currentText()
        dtaRng = self.window.calcDataIndicesFromLines(self.window.DATASTRINGS[0], 0)
        bw = self.ui.bwBox.value()
        logScale = False if self.ui.scaleModeBox.currentText() == 'Linear' else True

        # Error checking for user parameters
        if self.checkParameters(fftInt, fftShift, bw, dtaRng[1]-dtaRng[0]) == False:
            return

        self.updateCalculations(plotType, fftInt, fftShift, dtaRng, bw, logScale)

    def computeMag(self, dtaRng):
        # Computes the magnitude of the vector for every index in dtaRng
        sI, eI = dtaRng
        numVals = eI - sI

        dstrs = [box.currentText() for box in self.ui.vectorBoxes]
        dtas = [self.window.getData(dstr, self.window.currentEdit) for dstr in dstrs]

        magDta = [np.sqrt(dtas[0][i] ** 2 + dtas[1][i]**2 + dtas[2][i]**2) for i in range(sI, eI)]
        fullDta = np.empty(len(dtas[0]))
        fullDta[dtaRng[0]:dtaRng[1]] = magDta
        return list(fullDta)

    def updateCalculations(self, plotType, fftInt, fftShift, dtaRng, bw, logScale):
        self.ui.glw.clear() # Clear any previous grid elements

        # Get selected indices and times
        self.currIndices = dtaRng
        minIndex, maxIndex = dtaRng
        times = self.window.getTimes(self.window.DATASTRINGS[0], 0)[0]

        # Get full list of frequencies to be used when averaging by bandwidth
        freqs = self.calculateFreqList(1, fftInt)
        numFreqs = len(freqs)

        # Special vector magnitude pre-calculation for compress power
        magDta = None
        if plotType == 'Compressional Power':
            magDta = self.computeMag(dtaRng)

        # Calculate values in grid
        self.ui.statusBar.showMessage('Calculating...')
        timeStops = []
        valGrid = []
        startIndex, stopIndex = minIndex, minIndex + fftInt
        while stopIndex < maxIndex:
            timeStops.append(times[startIndex])
            dta = self.calcWaveAveraged(plotType, (startIndex, stopIndex), bw,
                numFreqs, magDta=magDta)
            valGrid.append(dta)
            startIndex += fftShift
            stopIndex = startIndex + fftInt

        # Add in last bounding time tick
        timeStops.append(times[startIndex])
        timeStops = np.array(timeStops)

        # Transpose to turn time result rows into columns
        valGrid = np.array(valGrid).T

        # Calculate frequencies for plotting and extra frequency for lower bound
        freqs = self.calculateFreqList(bw, fftInt)
        self.lastCalc = (timeStops, freqs, valGrid)
        diff = freqs[1] - freqs[0]
        if logScale and freqs[0] - diff <= 0:
            diff = diff * 0.5
        lowerBnd = freqs[0] - diff
        freqs = [lowerBnd] + list(freqs)

        defaultRng, gradStr, gradUnits = self.defParams[plotType]

        # Determine color gradient range
        logColorScale = True if plotType in self.plotGroups['Power'] else False
        if self.ui.selectToggle.isChecked():
            minVal, maxVal = self.ui.valueMin.value(), self.ui.valueMax.value()
            if logColorScale:
                minVal = 10 ** minVal
                maxVal = 10 ** maxVal
            colorRng = (minVal, maxVal)
        else: # Color range not set by user
            if logColorScale:
                colorRng = (np.min(valGrid[valGrid>0]), np.max(valGrid[valGrid>0]))
            else:
                colorRng = (np.min(valGrid), np.max(valGrid))
            colorRng = defaultRng if defaultRng is not None else colorRng

        # Generate spectrogram from set parameters and value grid
        self.ui.statusBar.showMessage('Generating plot...')
        plt = SpectrogramPlotItem(self.window.epoch, logScale)
        if plotType in self.plotGroups['Angle'] and colorRng == (-180, 180):
            plt = PhaseSpectrogram(self.window.epoch, logScale)
        plt.createPlot(freqs, valGrid, timeStops, colorRng, logColorScale=logColorScale,
            winFrame=self)

        # Set axis labels
        plt.setTitle(self.titleDict[plotType], size='13pt')
        yPrefix = 'Log ' if logScale else ''
        plt.getAxis('left').setLabel(yPrefix + 'Frequency (Hz)')
        self.ui.glw.addItem(plt, 0, 0, 1, 1)

        # Add in color gradient
        grad = plt.getGradLegend(logColorScale, offsets=(31, 47))
        grad.updateWidth(35)
        grad.setFixedWidth(65)
        self.ui.glw.nextCol()
        self.ui.glw.addItem(grad)
    
        # Add in gradient label
        if gradUnits is None:
            gradLbl = StackedAxisLabel([gradStr])
        else:
            gradLbl = StackedAxisLabel([gradStr, '['+gradUnits+']'])
        self.ui.glw.nextCol()
        self.ui.glw.addItem(gradLbl)
        gradLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        # Add in time information at bottom
        self.ui.addTimeInfo((timeStops[0], timeStops[-1]), self.window)

        self.ui.statusBar.clearMessage()

    def checkIndices(self):
        # Reset dictionaries if indices changed
        if self.currIndices is None or self.currIndices != self.prevIndices:
            self.prevIndices = self.currIndices
            self.fftDict = {}
            self.avgDict = {}

    def getAvg(self, dstr, en, iO, iE):
        self.checkIndices()

        # Calculate average or get from dictionary
        if dstr not in self.avgDict.keys():
            self.avgDict[dstr] = {}
        if (iO, iE) not in self.avgDict[dstr].keys():
            dta = self.window.getData(dstr, en)
            avg = np.mean(dta)
            self.avgDict[dstr][(iO, iE)] = avg
            return avg
        else:
            return self.avgDict[dstr][(iO, iE)]

    def getfft(self, dstr, en, iO, iE):
        self.checkIndices()

        # Calculate fft or get data from dictionary
        if dstr not in self.fftDict.keys():
            self.fftDict[dstr] = {}
        if (iO, iE) not in self.fftDict[dstr].keys():
            res = DynamicAnalysisTool.getfft(self, dstr, en, iO, iE)
            self.fftDict[dstr][(iO, iE)] = res
            return res
        else:
            return self.fftDict[dstr][(iO, iE)]

    def getNorm(self, v):
        # Faster than np.linalg.norm
        return np.sqrt(np.dot(v,v))

    def calcWaveAveraged(self, plotType, dtaRng, bw, numFreqs, magDta=None):
        halfBw = int((bw-1)/2) # Num of frequencies on either side of freqNum

        # Calculate the bw-averaged wave parameter over the data range
        sumMats = self.getAvgMats(dtaRng, numFreqs, bw, magDta)
        averagedDta = self.calcWave(plotType, (halfBw, numFreqs-halfBw), dtaRng, 
            bw, sumMats)

        return averagedDta

    def getAvgMats(self, dtaRng, numFreqs, bw, magDta=None):
        # Computes all the bw-averaged spectral mats for every valid freq index
        minIndex, maxIndex = dtaRng
        halfBw = int((bw-1)/2)

        en = self.window.currentEdit
        dstrs = [box.currentText() for box in self.ui.vectorBoxes]
        ffts = [self.getfft(dstr,en, minIndex, maxIndex) for dstr in dstrs]

        # Also pre-computes the magnitude fft used to calculate compressional
        # power if needed
        if magDta is not None:
            if '_Avg' not in self.fftDict.keys():
                self.fftDict['_Avg'] = {}
            subsetDta = magDta[minIndex:maxIndex]
            fft = fftpack.rfft(subsetDta)
            self.fftDict['_Avg'][dtaRng] = fft

        # Complex version of calculations
        cffts = [self.fftToComplex(fft) for fft in ffts]
        individualMats = self.computeSpectralMats(cffts)

        # Compute first sum up to halfBw
        currSum = np.zeros((3, 3), dtype=np.complex)
        for i in range(0, halfBw+halfBw+1):
            currSum += individualMats[i]

        # Compute a running sum (equiv. to sum(individualMats[i-halfBw:i+halfBw+1]))
        mats = np.empty((numFreqs, 3, 3), dtype=np.complex)
        mats[halfBw] = currSum
        for i in range(halfBw+1, numFreqs-halfBw):
            currSum = currSum - individualMats[i-halfBw-1] + individualMats[i+halfBw]
            mats[i] = currSum
        return mats

    def computeMinVarAngle(self, amat, avg):
        # Compute theta BK
        evn = np.array(amat[:,0])
        evi = np.array(amat[:,1])
        evx = np.array(amat[:,2])

        q = np.dot(avg, evn)

        if q < 0:
            evn = -1*evn
            evi = -1*evi
            amat[:,1] = -1*amat[:,1]
            amat[:,2] = -1*amat[:,2]

        if evx[2] < 0:
            evx = -1*evx
            evi = -1*evi
            amat[:,0] = -1*amat[:,0]
            amat[:,1] = -1*amat[:,1]

        evc = np.cross(evx, evi)
        q = np.dot(evc, evn)

        if q < 0:
            evi = -1*evi
            amat[:,1] = -1*amat[:,1]

        q = np.dot(evn, avg)
        vetm = np.dot(avg, avg)
        if vetm < 0:
            thbk = 0
        else:
            norm = np.sqrt(vetm)
            thbk = Mth.R2D * math.acos(q/norm)
        return thbk

    def calcWave(self, plotType, indexRange, dtaRng, bw, sumMats):
        minIndex, maxIndex = dtaRng
        numPoints = maxIndex - minIndex
        valLst = []

        numPairs = [(0, 1), (0, 2), (1, 2)] # Matrix indices used in calculations
        deltaf = (2.0 * self.window.resolution) / (numPoints * bw)

        if plotType in ['Ellipticity (Means)', 'Azimuth Angle',
            'Propagation Angle (Means)', 'Propagation Angle (BK)']:
            # Compute the average field for each dstr within the given time range
            avg = []
            for dstr in [box.currentText() for box in self.ui.vectorBoxes]:
                sI, eI = dtaRng
                dstrAvg = self.getAvg(dstr, self.window.currentEdit, sI, eI)
                avg.append(dstrAvg)
        
        if plotType == 'Compressional Power':
            # Computes the compressional power all at once
            halfBw = int((bw-1)/2)
            fftLst = self.fftDict['_Avg'][dtaRng]
            fftReal, fftImag = self.splitfft(fftLst)
            fftDouble = ((fftReal[:len(fftImag)] ** 2) + (fftImag ** 2)) * deltaf
            powerSum = []
            for i in range(indexRange[0], indexRange[1]):
                avgSum = 0
                for subIndex in range(i-halfBw, i+halfBw+1):
                    avgSum += fftDouble[subIndex] # fftReal^2 + fftImag^2
                powerSum.append(np.abs(avgSum))
            return powerSum

        for index in range(indexRange[0], indexRange[1]):
            # Get the pre-computed averaged cospectral matrix
            sumMat = sumMats[index] * deltaf

            # Cospectrum matrix is the real part of the matrix
            # Quadrature matrix is the imaginary part of the matrix
            realPower = sumMat.real
            imagPower = sumMat.imag

            # Extract info needed for wave propagation direction & angle calculations
            qs = [imagPower[i][j] for i,j in numPairs]
            cs = [realPower[i][j] for i,j in numPairs]

            prec = 7

            if plotType in self.plotGroups['Power']:
                pwspectra = np.abs(np.trace(realPower))
                valLst.append(pwspectra)
                continue

            if plotType in ['Ellipticity (Means)', 'Azimuth Angle',
                'Propagation Angle (Means)']:
                # Wave propogation direction
                qqq = self.getNorm(qs)
                qkem = np.array([qs[2] / qqq, -qs[1] / qqq, qs[0] / qqq])

                qqqp = self.getNorm(avg)
                qqqn = np.dot(qkem, avg)
                if qqqn < 0:
                    qkem = qkem * -1
                    qqqn = -1 * qqqn
                qqq = qqqn / qqqp
                qtem = Mth.R2D * math.acos(qqq) # field angle
                qqq = self.getNorm(cs)
                qdlm = np.array(cs[::-1] / qqq)
                qqqn = np.dot(qdlm, avg)
                qqq = qqqn / qqqp
                qalm = Mth.R2D * math.acos(qqq)

                if plotType == 'Propagation Angle (Means)':
                    valLst.append(qtem)
                    continue

                # Means transformation matrix
                yx = qkem[1] * avg[2] - qkem[2] * avg[1]
                yy = qkem[2] * avg[0] - qkem[0] * avg[2]
                yz = qkem[0] * avg[1] - qkem[1] * avg[0]
                qyxyz = self.getNorm([yx, yy, yz])
                yx = yx / qyxyz
                yy = yy / qyxyz
                yz = yz / qyxyz
                xx = yy * qkem[2] - yz * qkem[1]
                xy = yz * qkem[0] - yx * qkem[2]
                xz = yx * qkem[1] - yy * qkem[0]

                bmat = np.array([[xx, xy, xz], [yx, yy, yz], qkem])
                tmat = Mth.arpat(bmat, sumMat)
                trm = tmat.real # transformed real matrix
                tim = tmat.imag # transformed imaginary matrix
                elip, azim = self.joeMeansElip(trm, tim)
                if plotType == 'Ellipticity (Means)':
                    valLst.append(elip)
                    continue
                else:
                    valLst.append(azim)
                    continue
            else:
                duhh, amat = np.linalg.eigh(realPower, UPLO="U")
                amat = np.transpose(amat)
                if plotType == 'Propagation Angle (BK)':
                    bk = self.computeMinVarAngle(amat, avg)
                    valLst.append(bk)
                    continue

                # Transformed Values Spectral Matrices 
                tmat = Mth.arpat(amat, sumMat)
                tmat = Mth.flip(tmat)
                trp = tmat.real
                tip = tmat.imag
                elip = self.bornWolfElip(trp, tip)
                valLst.append(elip)
                continue
        return valLst

    def bornWolfElip(self, trp, tip):
        # Calculate polarization/ellipticity parameters by Born-Wolf
        trj = trp[0][0] + trp[1][1]
        detj = trp[0][0] * trp[1][1] - trp[1][0] * trp[1][0] - tip[1][0] * tip[1][0]

        vetm = trj * trj - 4.0 * detj
        eden = 1 if vetm <= 0 else math.sqrt(vetm)
        fnum = 2 * tip[0][1] / eden
        if (trp[0][1] < 0):
            elip = -1.0*math.tan(0.5*math.asin(fnum))
        else:
            elip = math.tan(0.5*math.asin(fnum))
        return abs(elip)

    def joeMeansElip(self, trm, tim):
        """
		Given transformed versions of real and imaginary powers and matrices
		calculate polarization and ellipticity (both with joe means versions), and azimuth angle
		"""
        # Calculate polarization/ellipticity parameters by Joe Means method
        trj = trm[0][0] + trm[1][1]
        detj=trm[0][0]*trm[1][1]-trm[0][1]*trm[1][0]-tim[1][0]*tim[1][0]

        vetm = trj * trj - 4.0 * detj
        eden = 1 if vetm <= 0 else math.sqrt(vetm)
        fnum = 2.0 * tim[0][1] / eden
        elipm = math.tan(0.5 * math.asin(fnum))

        # Calculate azimuth angle
        fnum = 2.0 * trm[0][1]
        difm = trm[0][0] - trm[1][1]
        angle = fnum / difm
        azim = 0.5 * math.atan(angle) * Mth.R2D
        return elipm, azim

    def getAxesStrs(self, dstrs):
        # Try to find variables matching the 'X Y Z' variable naming convention
        axisKws = ['X','Y','Z']
        axisDstrs = [[], [], []]

        kwIndex = 0
        for kw in axisKws:
            for dstr in dstrs:
                if kw.lower() in dstr.lower():
                    axisDstrs[kwIndex].append(dstr)
            kwIndex += 1
        return axisDstrs

    def showPointValue(self, freq, time):
        plotType = self.ui.waveParam.currentText()
        rng, gradStr, gradUnits = self.defParams[plotType]
        if gradStr == 'Log Power':
            gradStr = 'Power'
        self.showValue(freq, time, 'Freq, '+gradStr+': ', self.lastCalc)

    def showPreSelectWin(self):
        self.preWindow = PreDynWave(self)
        self.preWindow.show()
    
    def closePreSelectWin(self):
        if self.preWindow:
            self.preWindow.close()
            self.preWindow = None

class PreDynWaveUI(BaseLayout):
    def setupUI(self, winFrame, dynWindow):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        winFrame.setWindowTitle('Wave Analysis Parameters')
        winFrame.resize(100, 100)
        winFrame.move(0, 0)
        layout = QtWidgets.QGridLayout(winFrame)

        # Set up help text
        helpTxt = 'Pre-select the following parameters and then select a time region:'
        helpLbl = QtWidgets.QLabel(helpTxt)
        layout.addWidget(helpLbl, 0, 0, 1, 4)

        self.waveParam = QtWidgets.QComboBox()
        self.waveParam.addItems(dynWindow.defParams.keys())
        self.waveParam.setCurrentIndex(1)

        # Set up vector combo boxes
        self.vectorBoxes = []
        vecLt = QtWidgets.QHBoxLayout()
        for i in range(0, 3):
            box = QtWidgets.QComboBox()
            vecLt.addWidget(box)
            self.vectorBoxes.append(box)

        # Initialize default vector selection
        allDstrs = dynWindow.window.DATASTRINGS[:]
        axesStrs = dynWindow.getAxesStrs(allDstrs)
        if Mth.flattenLst(axesStrs, 1) == []:
            axesStrs = [allDstrs, allDstrs, allDstrs]

        for axLst, box in zip(axesStrs, self.vectorBoxes):
            box.addItems(axLst)
            if axLst == allDstrs:
                box.setCurrentIndex(self.vectorBoxes.index(box))

        # Set up frequency scaling and bandwidth boxes
        self.scaleModeBox = QtWidgets.QComboBox()
        self.scaleModeBox.addItems(['Linear', 'Logarithmic'])

        self.bwBox = QtWidgets.QSpinBox()
        self.bwBox.setMinimum(1)
        self.bwBox.setSingleStep(2)
        self.bwBox.setValue(3)

        self.addPair(layout, 'Plot Type: ', self.waveParam, 1, 0, 1, 1)
        self.addPair(layout, 'Frequency Scale: ', self.scaleModeBox, 1, 3, 1, 1)

        for row in range(0, 2):
            spcr = self.getSpacer(5)
            layout.addItem(spcr, row, 2)

        vecLbl = QtWidgets.QLabel('Vector: ')
        layout.addWidget(vecLbl, 2, 0, 1, 1)
        layout.addLayout(vecLt, 2, 1, 1, 1)
        self.addPair(layout, 'Bandwidth: ', self.bwBox, 2, 3, 1, 1)

        # Keeps window on top of main window while user updates lines
        winFrame.setParent(dynWindow.window)
        dialogFlag = QtCore.Qt.Dialog
        if dynWindow.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = winFrame.windowFlags()
        flags = flags | dialogFlag
        winFrame.setWindowFlags(flags)

        return winFrame

class PreDynWave(QtWidgets.QFrame, PreDynWaveUI):
    def __init__(self, mainWindow, parent=None):
        super(PreDynWave, self).__init__(parent)
        self.ui = PreDynWaveUI()
        self.ui.setupUI(self, mainWindow)

    def getParams(self):
        # Extract parameters from user interface
        plotType = self.ui.waveParam.currentText()
        vectorDstrs = [box.currentText() for box in self.ui.vectorBoxes]
        scaling = self.ui.scaleModeBox.currentText()
        bw = self.ui.bwBox.value()
        return (plotType, vectorDstrs, scaling, bw)
