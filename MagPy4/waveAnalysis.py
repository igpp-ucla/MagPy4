
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .dynBase import *
from .specAlg import SpectraCalc, WaveCalc
from .layoutTools import BaseLayout, VBoxLayout, HBoxLayout
from scipy import fftpack, signal
import pyqtgraph as pg
from .specAlg import SpecWave

import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
from .mth import Mth
import bisect
import math
from .MagPy4UI import MatrixWidget, NumLabel, StackedAxisLabel
import functools
import os

class VectorWidget(QtWidgets.QWidget):
    def __init__(self, vec_grps={}, init_vec=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUI()
        self.set_groups(vec_grps)
        if init_vec is not None:
            self.set_vector(init_vec)
    
    def setupUI(self):
        layout = HBoxLayout()
        for i in range(0, 3):
            box = QtWidgets.QComboBox()
            layout.addWidget(box)

        self.setLayout(layout)
    
    def _get_boxes(self):
        return self.layout().getItems()

    def get_vector(self):
        boxes = self._get_boxes()
        vec = [box.currentText() for box in boxes]
        return vec

    def set_vector(self, vec):
        boxes = self._get_boxes()
        for elem, box in zip(vec, boxes):
            box.setCurrentText(elem)

    def set_groups(self, vec_grps):
        boxes = self._get_boxes()
        for key in vec_grps:
            vec = vec_grps[key]
            for elem, box in zip(vec, boxes):
                box.addItem(elem)

class FrequencyWidget(QtWidgets.QWidget):
    def __init__(self, index_rng, freq_rng=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUI()
        self.set_index_range(index_rng)

    def setupUI(self):
        layout = HBoxLayout()
        self.setLayout(layout)

        self.minBox = QtWidgets.QSpinBox()
        self.maxBox = QtWidgets.QSpinBox()

        for box in [self.minBox, self.maxBox]:
            layout.addWidget(box)

        layout.addStretch()

    def set_index_range(self, rng):
        low, high = rng
        for box in [self.minBox, self.maxBox]:
            box.setMinimum(low)
            box.setMaximum(high)
    
        self.set_index_value(rng)

    def set_index_value(self, vals):
        low, high = vals
        self.minBox.setValue(low)
        self.maxBox.setValue(high)
    
    def get_index_range(self):
        low = self.minBox.value()
        high = self.maxBox.value()
        return (low, high)

class NumberText(QtWidgets.QLabel):
    def get_text(val, prec, complex_num=False):
        if val is None:
            return ''

        kwargs = {
            'precision' : prec,
            'unique' : False,
            'trim' : '0'
        }
        if not complex_num:
            txt = np.format_float_positional(val, **kwargs)
        else:
            real, imag = val.real, val.imag
            real_txt = np.format_float_positional(real, **kwargs)
            imag_txt = np.format_float_positional(imag, **kwargs)
            txt = f'{real_txt} + {imag_txt}j'
        
        return txt

class NumberLabel(QtWidgets.QLabel):
    def __init__(self, value=0, prec=5, imag=False, **kwargs):
        self.value = value
        self.prec = prec
        self.imag = imag

        super().__init__('', **kwargs)
        flags = QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse
        self.setTextInteractionFlags(flags)
        font = QtGui.QFont('monospace')
        self.setFont(font)

        self.set_value(value)

    def set_value(self, value):
        self.value = value
        txt = NumberText.get_text(value, self.prec, self.imag)
        self.setText(txt)
    
    def set_complex(self, val=True):
        self.imag = val

    def set_prec(self, prec):
        self.prec = prec
        self.set_value(self.value)

class VectorLabel(QtWidgets.QLabel):
    def __init__(self, vector=[0, 0, 0], prec=5, imag=False, **kwargs):
        self.prec = prec
        self.imag = imag

        super().__init__('', **kwargs)
        flags = QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse
        self.setTextInteractionFlags(flags)
        font = QtGui.QFont('monospace')
        self.setFont(font)

        self.set_vector(vector)

    def set_vector(self, v):
        self.vector = v

        txt_lbls = []
        for value in v:
            txt = NumberText.get_text(value, self.prec, self.imag)
            txt_lbls.append(txt)
        
        txt = '\n\n'.join(txt_lbls)
        self.setText(txt)
    
    def set_complex(self, val=True):
        self.imag = val

    def set_prec(self, prec):
        self.prec = prec
        self.set_value(self.vector)

class MatrixLabel(QtWidgets.QLabel):
    def __init__(self, mat=None, prec=5, imag=False, **kwargs):
        self.prec = prec
        self.imag = imag

        super().__init__('', **kwargs)
        flags = QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse
        self.setTextInteractionFlags(flags)
        font = QtGui.QFont('monospace')
        self.setFont(font)

        if mat is None:
            mat = np.eye(3)
        self.set_mat(mat)
    
    def set_mat(self, mat):
        self.mat = mat
        label_rows = []
        max_len = 0
        for row in mat:
            row_txt = []
            for elem in row:
                txt = NumberText.get_text(elem, self.prec, self.imag)
                row_txt.append(txt)
                max_len = max(max_len, len(txt))
            label_rows.append(row_txt)
        
        lines = []
        for row in label_rows:
            elems = [elem.ljust(max_len+2) for elem in row[:-1]]
            elems.append(row[-1])
            line = ''.join(elems)
            lines.append(line)
        
        txt = '\n\n'.join(lines)
        self.setText(txt)
        return txt

    def set_prec(self, prec):
        self.prec = prec
        self.set_value(self.mat)

    def set_complex(self, val=True):
        self.imag = val

class WaveAnalysisUI(object):
    def setupUI(self, Frame, window):
        Frame.setObjectName('waveFrame')
        style = '#waveFrame { background-color: white; }'
        Frame.setStyleSheet(style)
        Frame.setWindowTitle('Wave Analysis')
        Frame.resize(700,500)  

        # Set up vector options box
        ## Determine vector groupings
        grps = window.VECGRPS
        if len(grps) == 0:
            grps = {}
            dstrs = window.DATASTRINGS
            for i in range(0, len(dstrs)):
                dstr = dstrs[i]
                grps[i] = [dstr] * 3
            init_vec = dstrs[:3]
        else:
            init_vec = None

        ## Create copy to clipboard button
        self.clipboardBtn = QtWidgets.QPushButton('Copy to Clipboard')

        ## Set up widgets
        vec_layout = HBoxLayout()
        label = QtWidgets.QLabel('X, Y, Z:')
        self.vecWidget = VectorWidget(grps, init_vec=init_vec)
        vec_layout.addWidget(label)
        vec_layout.addWidget(self.vecWidget)
        vec_layout.addStretch()
        vec_layout.addWidget(self.clipboardBtn)

        # Set up frequency widget and update layout
        freq_layout = HBoxLayout()
        label = QtWidgets.QLabel('Frequency Range:')
        self.freqWidget = FrequencyWidget((0, 20))

        self.updateBtn = QtWidgets.QPushButton('Update')

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizeGripEnabled(False)
        self.statusBar.setMinimumWidth(175)

        for item in [label, self.freqWidget, self.updateBtn]:
            freq_layout.addWidget(item)
        freq_layout.addStretch(1)
        freq_layout.addWidget(self.statusBar)

        # Set up matrix info boxes
        layout = QtWidgets.QGridLayout()
        prec = 5
        self.matlabels = {}
        keys = [' Power', 'Transformed Power', 'Transformed Matrix']
        row = 0
        for key in keys:
            # Create matrix label objects
            real = MatrixLabel(prec=prec)
            imag = MatrixLabel(prec=prec)
            labels = [real, imag]

            # Wrap labels in a groupbox frame
            box_kws = ['Real', 'Imaginary']
            for i in range(0, 2):
                # Set up frame label
                mat_type = box_kws[i]

                box_label = key.split(' ')
                box_label.insert(1, mat_type)
                box_label = ' '.join(box_label).strip(' ')

                # Add frame to outer layout
                frame = QtWidgets.QGroupBox(box_label)
                frm_lt = QtWidgets.QVBoxLayout(frame)
                frm_lt.addWidget(labels[i])
                layout.addWidget(frame, row, i, 1, 1)

            # Store boxes and update row
            key = key.strip(' ')
            self.matlabels[key] = labels
            row += 1

        # Set up eigenvector and eigenvalue boxes
        self.eigenvecbox = MatrixLabel(prec=prec)
        self.eigenvalbox = VectorLabel(prec=prec)

        row = 0
        labels = ['Eigenvectors [v1 | v2 | v3]', 'Eigenvalues [λ1, λ2, λ3]']
        for i, vec in enumerate([self.eigenvecbox, self.eigenvalbox]):
            # Create wrapper frame
            label = labels[i]
            frame, frm_lt = self.wrap_widget(vec, label, center=True)

            # Add to outer layout
            layout.addWidget(frame, row, 2+i, 1, 1)
        
        # Set up propagation and linear variance vectors and angles
        prop_layout = QtWidgets.QGridLayout()

        self.propvecbox = VectorLabel(prec=prec)
        self.linvarvecbox = VectorLabel(prec=prec)

        self.propanglelbl = NumberLabel(prec=prec)
        self.linvaranglelbl = NumberLabel(prec=prec)

        ## Vector wrapper frames
        row = 0
        labels = ['Propagation Vec', 'Linear Var Vec']
        for i, vec in enumerate([self.propvecbox, self.linvarvecbox]):
            label = labels[i]
            frame, frm_lt = self.wrap_widget(vec, label, center=True)
            prop_layout.addWidget(frame, row, i, 2, 1)
        
        ## Angle wrapper frames
        col = 2
        labels = ['Propagation Angle', 'Linear Var Angle']
        for i, angle in enumerate([self.propanglelbl, self.linvaranglelbl]):
            label = labels[i]
            frame, frm_lt = self.wrap_widget(angle, label, center=True)
            prop_layout.addWidget(frame, row+i, col)

        ## Add prop layout to main layout
        layout.addLayout(prop_layout, 1, 2, 1, 2)

        # Set up ellipticity, percent polarization, and azimuth angle frame
        ellip_layout = QtWidgets.QGridLayout()

        ## Method labels
        left_label = '\n'.join(['', '  Born-Wolf:', '', '  Joe-Means:'])
        left_label = QtWidgets.QLabel(left_label)
        left_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        ## Ellipticity
        self.ellip_box = VectorLabel([0, 0], prec=prec)
        lbl = 'Ellipticity'
        ellip_frm, lt = self.wrap_widget(self.ellip_box, lbl, True)

        ## Percent polarization
        self.perc_pol_box = VectorLabel([0,0], prec=prec)
        lbl = '% Polarization'
        perc_pol_frm, lt = self.wrap_widget(self.perc_pol_box, lbl, True)

        ## Azimuthal angle
        self.azimuth_label = VectorLabel([0, None], prec=prec)
        lbl = 'Azimuth Angle'
        azimuth_frm, lt = self.wrap_widget(self.azimuth_label, lbl, True)

        ellip_layout.addWidget(left_label, 0, 0, 1, 1)
        ellip_layout.addWidget(ellip_frm, 0, 1, 1, 1)
        ellip_layout.addWidget(perc_pol_frm, 0, 2, 1, 1)
        ellip_layout.addWidget(azimuth_frm, 0, 3, 1, 1)
        layout.addLayout(ellip_layout, 2, 2, 1, 2)

        wrap_layout = VBoxLayout(Frame)
        wrap_layout.addLayout(vec_layout)
        wrap_layout.addLayout(layout)
        wrap_layout.addLayout(freq_layout)

    def wrap_widget(self, widget, label, center=False):
        frame = QtWidgets.QGroupBox(label)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.addWidget(widget)
    
        if center:
            frame.setAlignment(QtCore.Qt.AlignHCenter)
            layout.setAlignment(QtCore.Qt.AlignHCenter)

        return frame, layout

    def getItems(self):
        items = []
        for key in self.matlabels:
            matitems = self.matlabels[key]
            items.extend(matitems)
        
        other_items = [self.eigenvecbox, self.eigenvalbox, self.propvecbox, 
            self.linvarvecbox,  self.propanglelbl, self.linvaranglelbl, 
            self.ellip_box, self.perc_pol_box, self.azimuth_label]
        
        return items + other_items

class WaveAnalysis(QtWidgets.QFrame, WaveAnalysisUI):
    def __init__(self, spectra, window, parent=None):
        super(WaveAnalysis, self).__init__(parent)

        self.spectra = spectra
        self.window = window
        self.ui = WaveAnalysisUI()
        self.ui.setupUI(self, window)

        self.ui.updateBtn.clicked.connect(self.updateCalculations)
        self.ui.clipboardBtn.clicked.connect(self.copyToClipboard)

        freqs = self.getDefaultFreqs()
        m = len(freqs)
        self.ui.freqWidget.set_index_range((0, m-1))

        self.updateCalculations() # should add update button later

    def copyToClipboard(self):
        ''' Copys window text to clipboard '''

        # Get UI items and assemble text from their values
        items = self.ui.getItems()
        text = ''
        for item in items:
            # Get item label
            box = item.parent()
            label = box.title()

            # Get item value and trim double newlines
            value = item.text().replace('\n\n', '\n')

            # Add to text
            text = f'{text}\n\n{label}\n{value}'

        # Create clipboard and add text to it
        clipboard = QtGui.QGuiApplication.clipboard()
        clipboard.setText(text)

        # Reflect in status bar
        self.ui.statusBar.showMessage('Successfully copied...', 2000)

    def getDefaultFreqs(self):
        dstrs = self.ui.vecWidget.get_vector()
        en = self.window.currentEdit
        sI, eI = self.spectra.getIndices(dstrs[0], en)
        times, diff, res = self.spectra.window.getTimes(dstrs[0], en)
        n = abs(eI -sI)
        return SpectraCalc.calc_freq(1, n, res)

    def updateCalculations(self):
        """ Update all wave analysis values and corresponding UI elements """
        en = self.window.currentEdit
        dstrs = self.ui.vecWidget.get_vector()
        ffts = [self.spectra.getfft(dstr,en) for dstr in dstrs]

        fO, fE = self.ui.freqWidget.get_index_range()

        # ensure first frequency index is less than last and not the same value
        if fE < fO:
            fO,fE = fE,fO
        if abs(fO-fE) < 2:
            fE = fO + 2
        self.ui.freqWidget.set_index_value((fO, fE))

        # Compute the average field for each dstr within the given time range
        data = []
        times, diffs, res = self.window.getTimes(dstrs[0], en)
        for dstr in dstrs[0:3]:
            sI, eI = self.spectra.getIndices(dstr, en)
            data.append(self.window.getData(dstr, en)[sI:eI])
        avg = np.mean(np.vstack(data), axis=1)
        
        params = {'num_points' : eI-sI, 'resolution':res,
            'freq_range' : (fO, fE)}
        sw = SpecWave()
        params = sw.get_params(ffts, params, avg)

        # Update matrix information
        label_kws = ['Power', 'Transformed Power', 'Transformed Matrix']
        spec_kws = ['sum_mat', 'bw_transf_pow', 'means_transf_mat']
        for label_kw, spec_kw in zip(label_kws, spec_kws):
            real_label, imag_label = self.ui.matlabels[label_kw]
            mat = params[spec_kw]
            real_label.set_mat(mat.real)
            imag_label.set_mat(mat.imag)

        # Update eigenvectors and eigenvalues
        evals = params['bw_eigenvalues']
        evectors = params['bw_eigenvectors']
        self.ui.eigenvecbox.set_mat(evectors)
        self.ui.eigenvalbox.set_vector(evals)

        # Update propagation vector and angle
        propvec = params['means_k']
        propangle = params['means_prop_angle']
        self.ui.propvecbox.set_vector(propvec)
        self.ui.propanglelbl.set_value(propangle)

        # Update linear variance vector and angle
        linvarvec = params['lin_var_vec']
        linvarangle = params['lin_var_angle']
        self.ui.linvarvecbox.set_vector(linvarvec)
        self.ui.linvaranglelbl.set_value(linvarangle)

        # Update ellipticity values
        bw_ellip = params['bw_ellip']
        means_ellip = params['means_ellip']
        self.ui.ellip_box.set_vector([bw_ellip, means_ellip])

        # Update percent polarization values
        bw_perc_pol = params['bw_perc_polar']
        means_perc_pol = params['means_perc_polar']
        self.ui.perc_pol_box.set_vector([bw_perc_pol, means_perc_pol])

        # Update azimuth angle
        azimuth = params['means_azim_angle']
        self.ui.azimuth_label.set_vector([azimuth, None])

class DynamicWaveUI(BaseLayout):
    def setupUI(self, Frame, window, params):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Wave Analysis')
        Frame.resize(800, 775)
        layout = QtWidgets.QGridLayout(Frame)

        # Create additional widget for adding plots to main window
        self.addBtn = Frame.getAddBtn()
        if isinstance(window, QtGui.QMainWindow):
            widgets = [self.addBtn]
        else:
            widgets = []

        # Setup time edits / status bar and the graphics frame
        lt, self.timeEdit, self.statusBar = self.getTimeStatusBar(widgets)
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
        times = window.getTimes(self.vectorBoxes[0].currentText(), window.currentEdit)[0]
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

        # Setup data points indicator and detrend checkbox
        self.fftDataPts = QtWidgets.QLabel()
        ptsTip = 'Total number of data points within selected time range'

        self.detrendCheck = QtWidgets.QCheckBox(' Detrend Data')
        self.detrendCheck.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        detrendTip = 'Detrend data using the least-squares fit method before each FFT is applied '
        self.detrendCheck.setToolTip(detrendTip)

        ptsLt = QtWidgets.QGridLayout()
        ptsLt.addWidget(self.fftDataPts, 0, 0, 1, 1)
        spacer = self.getSpacer(10)
        ptsLt.addItem(spacer, 0, 1, 1, 1)
        ptsLt.addWidget(self.detrendCheck, 0, 2, 1, 1)

        ptsLbl = QtWidgets.QLabel('Data Points: ')
        ptsLbl.setToolTip(ptsTip)
        layout.addWidget(ptsLbl, 2, 0, 1, 1)
        layout.addLayout(ptsLt, 2, 1, 1, 1)

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
        self.addLineBtn = QtWidgets.QPushButton('Add Line')
        self.addLineBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.updtBtn = QtWidgets.QPushButton('Update')
        self.updtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.maskBtn = QtWidgets.QPushButton('Mask')
        self.maskBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        layout.addWidget(self.updtBtn, 2, 7, 1, 1)
        layout.addWidget(self.addLineBtn, 1, 7, 1, 1)
        layout.addWidget(self.maskBtn, 0, 7, 1, 1)

        return layout

    def setupFFTLayout(self):
        layout = QtWidgets.QGridLayout()
        row = 0
        # Set up fft parameter spinboxes
        self.fftShift = QtWidgets.QSpinBox()
        self.fftInt = QtWidgets.QSpinBox()

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

        # Multiprocessing parameters
        try:
            numProcs = multiprocessing.cpu_count()
        except:
            numProcs = 1
        if numProcs >= 4:
            self.numThreads = 4
        elif numProcs >= 2:
            self.numThreads = 2
        else:
            self.numThreads = 1
        
        # Disable multiprocessing for windows
        # TODO: Modify functions so calculations do not need to depend on main window
        if window.OS == 'windows':
            self.numThreads = 1

        self.mpPointBound = 5000 # Number of points needed to use multiprocessing

        # Default settings / labels for each plot type
        self.defParams = { # Value range, grad label, grad label units
            'Azimuth Angle' : ((-90, 90), 'Azimuth Angle', 'Degrees'),
            'Ellipticity (Means)' : ((-1.0, 1.0), 'Ellipticity', None),
            'Ellipticity (SVD)' : ((-1.0, 1.0), 'Ellipticity', None),
            'Ellipticity (Born-Wolf)' : ((0, 1.0), 'Ellipticity', None),
            'Propagation Angle (Means)' : ((0, 90), 'Angle', 'Degrees'),
            'Propagation Angle (SVD)' : ((0, 90), 'Angle', 'Degrees'),
            'Propagation Angle (Min Var)' : ((0, 90), 'Angle', 'Degrees'),
            'Power Spectra Trace' : (None, 'Log Power', 'nT^2/Hz'),
            'Compressional Power' : (None, 'Log Power', 'nT^2/Hz')
        }

        self.titleDict = {}
        for key in self.defParams.keys():
            self.titleDict[key] = key
        self.titleDict['Power Spectra Trace'] = 'Trace Power Spectral Density'
        self.titleDict['Propagation Angle (Min Var)'] = 'Minimum Variance Angle'
        self.titleDict['Compressional Power'] = 'Compressional Power Spectral Density'

        # Sorts plot type names into groups
        self.plotGroups = {'Angle' : [], 'Ellipticity' : [], 'Power' : []}
        for plotType in self.defParams.keys():
            for kw in self.plotGroups.keys():
                if kw in plotType:
                    self.plotGroups[kw].append(plotType)
                    break

        self.fftDict = {} # Stores dicts for each dstr, second key is indices
        self.avgDict = {}

        self.lastCalc = None # Stores last calculated times, freqs, values
        self.plotItem = None

        self.ui.setupUI(self, window, self.defParams.keys())
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.addLineBtn.clicked.connect(self.openLineTool)
        self.ui.maskBtn.clicked.connect(self.openMaskTool)
        self.ui.addBtn.clicked.connect(self.addToMain)

    def closeEvent(self, ev):
        self.close()
        self.closeLineTool()
        self.closeMaskTool()
        self.window.endGeneralSelect()
        if self.plotItem:
            self.plotItem.closePlotAppearance()

    def getToolType(self):
        return self.ui.waveParam.currentText()

    def getVarInfo(self):
        return [box.currentText() for box in self.ui.vectorBoxes]

    def setVarParams(self, varInfo):
        for box, var in zip(self.ui.vectorBoxes, varInfo):
            box.setCurrentText(var)

    def loadState(self, state):
        # Makes sure default selections do not override loaded state info
        self.closePreSelectWin()

        # Range settings are reset if plot type is set after loading rest of state
        self.setPlotType(state['plotType'])
        DynamicAnalysisTool.loadState(self, state)

    def setPlotType(self, pltType):
        self.ui.waveParam.setCurrentText(pltType)

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

    def getDataRange(self):
        dstr = self.ui.vectorBoxes[0].currentText()
        return self.window.calcDataIndicesFromLines(dstr, 0)

    def getGradRange(self):
        if self.ui.selectToggle.isChecked():
            minVal = self.ui.valueMin.value()
            maxVal = self.ui.valueMax.value()
            # Adjust order if flipped
            minVal, maxVal = min(minVal, maxVal), max(minVal, maxVal)
            if self.ui.valueMin.prefix() == '10^':
                minVal = 10 ** minVal
                maxVal = 10 ** maxVal
            gradRange = (minVal, maxVal)
        else:
            gradRange = None
        return gradRange

    def checkIfPlotTypeChanged(self):
        # If plot type is changed and a mask tool is open,
        # close the mask tool
        if self.maskTool:
            maskType = self.maskTool.plotType
            if maskType != self.ui.waveParam.currentText():
                self.closeMaskTool()

    def update(self):
        self.checkIfPlotTypeChanged()
        fftInt, fftShift, bw = self.getFFTParam()
        plotType = self.ui.waveParam.currentText()
        dtaRng = self.getDataRange()
        logScale = False if self.getAxisScaling() == 'Linear' else True
        detrendMode = self.getDetrendMode()

        fftParam = (fftInt, fftShift, bw)
        vecDstrs = self.getVarInfo()

        # Error checking for user parameters
        if self.checkParameters(fftInt, fftShift, bw, dtaRng[1]-dtaRng[0]) == False:
            return

        # Calculate grid values and generate plot items
        grid, freqs, times = self.calcGrid(plotType, dtaRng, fftParam, vecDstrs, detrendMode)
        colorRng = self.getColorRng(plotType, grid)
        plt = self.generatePlots(grid, freqs, times, colorRng, plotType, logScale)
        self.setupPlotLayout(plt, plotType, times, logScale)

        # Save state
        self.plotItem = plt
        self.lastCalc = (times, freqs, grid)

        # Enable exporting plot data
        fftParam = (fftInt, fftShift, bw, detrendMode)
        exportFunc = functools.partial(self.exportData, self.window, plt, fftParam)
        self.plotItem.setExportEnabled(exportFunc)

        if self.savedLineInfo: # Add any saved lines
            self.addSavedLine()
        elif len(self.lineInfoHist) > 0 and len(self.lineHistory) == 0:
            self.savedLineInfo = self.lineInfoHist
            self.lineInfoHist = []
            self.addSavedLine()
            self.savedLineInfo = None

    def getLabels(self, plotType, logScaling):
        # Determine plot title and y axis label
        title = self.titleDict[plotType]

        axisLbl = 'Frequency (Hz)'
        if logScaling:
            axisLbl = 'Log ' + axisLbl

        # Get gradient legend label and units for this plot type        
        valRng, gradLbl, units = self.defParams[plotType]
        gradLblStrs = [gradLbl]
        if units:
            gradLblStrs.append('['+units+']')

        # Build gradient legend's stacked label and rotate if necessary
        angle = 90 if plotType in self.plotGroups['Ellipticity'] else 0
        legendLbl = StackedAxisLabel(gradLblStrs, angle=angle)
        legendLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        return title, axisLbl, legendLbl

    def getGradTickSpacing(self, plotType):
        if plotType in self.plotGroups['Angle']:
            tickSpacing = (60, 30)
        elif plotType in self.plotGroups['Ellipticity']:
            tickSpacing = (0.2, 0.1)
        else:
            tickSpacing = (1, 0.5)
        return tickSpacing

    def setupPlotLayout(self, plt, plotType, times, logMode):
        # Get gradient legend and set its tick spacing accordingly
        gradLegend = plt.getGradLegend(logMode=False)
        major, minor = self.getGradTickSpacing(plotType)
        gradLegend.setTickSpacing(major, minor)
        gradLegend.setBarWidth(40)

        # Get title, y axis, and gradient legend labels
        title, axisLbl, legendLbl = self.getLabels(plotType, logMode)
        plt.setTitle(title, size='13pt')
        plt.getAxis('left').setLabel(axisLbl)

        # Update specData information
        specData = plt.getSpecData()
        specData.set_name(title)
        specData.set_y_label(axisLbl)
        specData.set_legend_label(legendLbl.getLabelText())

        # Add in time range label at bottom
        timeInfo = self.getTimeInfoLbl((times[0], times[-1]))

        self.ui.glw.clear()
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.ui.glw.addItem(gradLegend, 0, 1, 1, 1)
        self.ui.glw.addItem(legendLbl, 0, 2, 1, 1)
        self.ui.glw.addItem(timeInfo, 1, 0, 1, 3)

    def getColorRng(self, plotType, grid):
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
                colorRng = (np.min(grid[grid>0]), np.max(grid[grid>0]))
            else:
                colorRng = (np.min(grid), np.max(grid))
            colorRng = defaultRng if defaultRng is not None else colorRng
        
        return colorRng

    def calcGrid(self, plotType, dtaRng, fftParam, vecDstrs, detrend=False):
        # Unpack parameters
        fftInt, fftShift, bw = fftParam
        minIndex, maxIndex = dtaRng

        # Get data and times for selected time region
        data = [self.window.getData(dstr, self.window.currentEdit) for dstr in vecDstrs]
        data = np.vstack(data)[:,minIndex:maxIndex]
        times, diff, res = self.window.getTimes(vecDstrs[0], self.window.currentEdit)
        times = times[minIndex:maxIndex]

        # Map plot type to compute function and any arguments needed
        func_map = {
            'Azimuth Angle' : (WaveCalc.calc_azimuth_angle, None),
            'Ellipticity (Means)' : (WaveCalc.calc_ellip, 'means'),
            'Ellipticity (SVD)' : (WaveCalc.calc_ellip, 'svd'),
            'Ellipticity (Born-Wolf)' : (WaveCalc.calc_ellip, 'bw'),
            'Propagation Angle (Means)' : (WaveCalc.calc_prop_angle, 'means'),
            'Propagation Angle (SVD)' : (WaveCalc.calc_prop_angle, 'svd'),
            'Propagation Angle (Min Var)' : (WaveCalc.calc_prop_angle, 'minvar'),
            'Power Spectra Trace' : (WaveCalc.calc_power_trace, None),
            'Compressional Power' : (WaveCalc.calc_compress_power, None),
        }

        func, arg = func_map[plotType]
        func_args = {}
        if arg is not None:
            func_args = {'method':arg}

        # Create fft_params dictionary to pass to compute function
        fft_params = {
            'bandwidth' : bw,
            'resolution' : res,
            'num_points' : fftInt,
        }

        # Split data into segments and get time stops for each section
        timeStops, dataSegs = self.splitDataSegments(times, data, fftInt, 
            fftShift)
        
        # Detrend data if necessary
        if detrend:
            new_segs = []
            for seg in dataSegs:
                seg = np.vstack([signal.detrend(row) for row in seg])
                new_segs.append(seg)
            dataSegs = new_segs

        # If large amount of data and number of threads is > 1, parallelize
        # computations
        if maxIndex - minIndex > 5000 and self.numThreads > 1:
            groups = ParallelGrid.create_groups(dataSegs, self.numThreads)
            valGrid = ParallelGrid.parallelize(groups, func, [fft_params], func_args)
        else:
            # Otherwise, compute each segment in order
            valGrid = []
            for sub_data in dataSegs:
                result = func(sub_data, fft_params, **func_args)
                valGrid.append(result)

        # Transpose to turn rows into columns
        valGrid = np.vstack(valGrid).T
        timeStops = np.array(timeStops)

        # Compute frequencies
        freqs = SpectraCalc.calc_freq(bw, fftInt, res)

        return valGrid, freqs, timeStops

    def generatePlots(self, grid, freqs, times, colorRng, plotType, logMode):
        freqs = self.extendFreqs(freqs, logMode)
        plt = SpectrogramPlotItem(self.window.epoch, logMode)
        logColorScale = True if plotType in self.plotGroups['Power'] else False
        plt.createPlot(freqs, grid, times, colorRng, winFrame=self,
            logColorScale=logColorScale)
        return plt

    def extendFreqs(self, freqs, logScale):
        # Calculate frequency that serves as lower bound for plot grid
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and logScale:
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])
        return freqs

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

    def addLineToPlot(self, line):
        self.plotItem.addItem(line)
        self.lineHistory.add(line)

    def removeLinesFromPlot(self):
        histCopy = self.lineHistory.copy()
        for line in histCopy:
            if line in self.plotItem.listDataItems():
                self.plotItem.removeItem(line)
                self.lineHistory.remove(line)
