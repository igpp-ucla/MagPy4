# FilterDialog.py - Filter dialog box
#
# The Filter dialog box user interface was designed using Qt Designer, which
# writes out the file FilterDialog.ui. This file is in turn converted into
# FilterDialogUI.py at the command-line using the command:
#
# pyuic5 FilterDialog.ui -o FilterDialogUI.py

# The following notes apply to some of the code below and to code in
# filter.py:
#
# Lawrence R. Rabiner (Bell Lab), Carol A. McGonegal (Bell Lab),
# and Douglas Paul (MIT Lincoln Lab), FIR Windowed Filter Design
# program - WINDOW, in "Programs for Digital Signal Processing,"
# 5.2-1 to 5.2-19, IEEE Press, New York, 1979.
#
# The subroutine was converted from banal FORTRAN code to C code
# by Gordon MaClean (IGPP).
#
# FORTRAN code header:
#
# C SUBROUTINE:   WINDOW DESIGN OF LINEAR PHASE, LOWPASS, HIGHPASS,
# C               BANDPASS, AND BANDSTOP FIR DIGITAL FILTERS
# C AUTHOR:       LAWRENCE R. RABINER AND CAROL A. MCGONEGAL
# C               BELL LABORATORIES, MURRAY HILL, NEW JERSEY, 07974
# C MODIFIED JAN. 1978 BY DOUG PAUL, MIT LINCOLN LABORATORIES
# C TO INCLUDE SUBROUTINES FOR OBTAINING FILTER BAND EDGES AND RIPPLES
# C MODIFIED MAR. 1982 BY NC TO TURN PROGRAM INTO SUBROUTINE
#
# Recoded to Python and PyQt 2014 May 07, by L. Lee

import numpy as np

from enum import Enum
from filter import chebyshevParameters
from filter import lowpassResponse
from filter import bandpassResponse
from filter import rectangularWindow
from filter import triangularWindow
from filter import hammingWindow
from filter import kaiserWindow
from filter import chebyshevWindow
from FilterDialogUI import Ui_FilterDialog
from mth import Mth
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

class FilterDialog(QtWidgets.QDialog, Ui_FilterDialog):

    def __init__(self, edit, parent=None):
        super(FilterDialog, self).__init__()

        self.ui = Ui_FilterDialog()
        self.ui.setupUi(self)

        # Parent is the program's main window. The variable 'window' is used for the
        # main window elsewhere in this program, but here I think it would be too
        # easily confused with the filter window. (fair point - jfc3)
        self.parent = parent
        self.edit = edit

        # Remove the question mark (context help button) from the dialog box's title bar.
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        # Wire up the combo boxes.
        self.ui.filterTypeComboBox.currentTextChanged.connect(self.onFilterTypeComboBoxChanged)
        self.ui.filterWidthComboBox.currentTextChanged.connect(self.onFilterWidthComboBoxChanged)
        self.ui.windowTypeComboBox.currentTextChanged.connect(self.onWindowTypeComboBoxChanged)

        # Wire up the OK and Cancel buttons.
        self.accepted.connect(self.onAccepted)
        self.rejected.connect(self.onRejected)

        # Combo box selected items
        # I tried using enums for this sort of stuff but it wasn't worth the effort.
        self.filterType = 'Low Pass'
        self.filterWidth = 'Points'
        self.windowType = 'Rectangular'

        # User-specified values
        self.lowCutoffFreq = 0.0
        self.highCutoffFreq = 0.0
        self.numPoints = 101
        self.oldNumPoints = 0
        self.halfNumPoints = int((self.numPoints + 1.0) / 2.0)
        self.timespan = 3.997
        self.alpha = 0.54
        self.stopBandAttenuation = 100.0
        self.ripple = 2.0
        self.transitionWidth = 0.0

        # Calculated values
        self.noff = 0
        self.nyquistFreq = 0.0

        # For now, use 60 as the resolution. Later change this to the data resolution.
        self.resolution = 60.0
        self.setResolution(self.resolution)
        # self.setWidgets()

        # Filter and window methods
        self.generateFilter = None
        self.generateWindow = None

        # Filter coefficients
        self.g = None
        self.w = None
        self.gw = None

        self.chebyshevParameters()
        self.setWidgets()

        self.updateFilterType()
        self.updateWindowType()

        self.displayNyquistFreq()

        # Initialize the combo boxes.
        self.onFilterTypeComboBoxChanged(self.filterType)
        self.onFilterWidthComboBoxChanged(self.filterWidth)
        self.onWindowTypeComboBoxChanged(self.windowType)

    # Dialog data exchange
    # values -> widgets
    # widgets -> values

    def setWidgets(self):
        """Set the widgets to the user-definable values.
        """
        self.ui.lowCutoffFreqText.setText(f'{self.lowCutoffFreq:.6f}')
        self.ui.highCutoffFreqText.setText(f'{self.highCutoffFreq:.6f}')
        self.ui.numPointsText.setText(f'{self.numPoints}')
        self.ui.timespanText.setText(f'{self.timespan}')
        self.ui.alphaText.setText(f'{self.alpha}')
        self.ui.stopBandAttenuationText.setText(f'{self.stopBandAttenuation}')
        self.ui.rippleText.setText(f'{self.ripple:.6f}')
        self.ui.transitionWidthText.setText(f'{self.transitionWidth:.6f}')

    def getValues(self):
        """Get the user-definable values from the widgets.
        """
        self.filterType = self.ui.filterTypeComboBox.currentText()
        self.lowCutoffFreq = float(self.ui.lowCutoffFreqText.text())
        self.highCutoffFreq = float(self.ui.highCutoffFreqText.text())
        self.filterWidth = self.ui.filterWidthComboBox.currentText()
        self.numPoints = int(self.ui.numPointsText.text())
        self.halfNumPoints = int((self.numPoints + 1.0) / 2.0)
        self.timespan = float(self.ui.timespanText.text())
        self.windowType = self.ui.windowTypeComboBox.currentText()
        self.alpha = float(self.ui.alphaText.text())
        self.stopBandAttenuation = float(self.ui.stopBandAttenuationText.text())
        self.ripple = float(self.ui.rippleText.text())
        self.transitionWidth = float(self.ui.transitionWidthText.text())

    # Methods to enable or disable groups of widgets
    # This would require less code if I used layouts, but I want more control over the placment of widgets.

    def setLowCutoffFreqEnabled(self, enabled):
        self.ui.lowCutoffFreqLabel.setEnabled(enabled)
        self.ui.lowCutoffFreqText.setEnabled(enabled)
        self.ui.lowCutoffFreqUnit.setEnabled(enabled)

    def setHighCutoffFreqEnabled(self, enabled):
        self.ui.highCutoffFreqLabel.setEnabled(enabled)
        self.ui.highCutoffFreqText.setEnabled(enabled)
        self.ui.highCutoffFreqUnit.setEnabled(enabled)

    def setNumPointsEnabled(self, enabled):
        self.ui.numPointsLabel.setEnabled(enabled)
        self.ui.numPointsText.setEnabled(enabled)

    def settimespanEnabled(self, enabled):
        self.ui.timespanLabel.setEnabled(enabled)
        self.ui.timespanText.setEnabled(enabled)
        self.ui.timespanUnit.setEnabled(enabled)

    def setAlphaEnabled(self, enabled):
        self.ui.alphaLabel.setEnabled(enabled)
        self.ui.alphaText.setEnabled(enabled)

    def setStopBandAttenuationEnabled(self, enabled):
        self.ui.stopBandAttenuationLabel.setEnabled(enabled)
        self.ui.stopBandAttenuationText.setEnabled(enabled)
        self.ui.stopBandAttenuationUnit.setEnabled(enabled)

    def setRippleEnabled(self, enabled):
        self.ui.rippleLabel.setEnabled(enabled)
        self.ui.rippleText.setEnabled(enabled)
        self.ui.rippleUnit.setEnabled(enabled)

    def setTransitionWidthEnabled(self, enabled):
        self.ui.transitionWidthLabel.setEnabled(enabled)
        self.ui.transitionWidthText.setEnabled(enabled)
        self.ui.transitionWidthUnit.setEnabled(enabled)

    def onFilterTypeComboBoxChanged(self, value):
        """Called when the user changes the filter type
        """
        if value == 'Low Pass':
            self.setLowCutoffFreqEnabled(False)
            self.setHighCutoffFreqEnabled(True)
        elif value == 'High Pass':
            self.setLowCutoffFreqEnabled(True)
            self.setHighCutoffFreqEnabled(False)
        elif value == 'Band Pass':
            self.setLowCutoffFreqEnabled(True)
            self.setHighCutoffFreqEnabled(True)
        elif value == 'Band Stop':
            self.setLowCutoffFreqEnabled(True)
            self.setHighCutoffFreqEnabled(True)

    def onFilterWidthComboBoxChanged(self, value):
        """Called when the user changes the filter width
        """
        if value == 'Points':
            self.setNumPointsEnabled(True)
            self.settimespanEnabled(False)
        elif value == 'Time':
            self.setNumPointsEnabled(False)
            self.settimespanEnabled(True)

    def onWindowTypeComboBoxChanged(self, value):
        """Called when the user changes the window type
        """
        if value == 'Rectangular':
            self.setAlphaEnabled(False)
            self.setStopBandAttenuationEnabled(False)
            self.setRippleEnabled(False)
            self.setTransitionWidthEnabled(False)
        elif value == 'Triangular':
            self.setAlphaEnabled(False)
            self.setStopBandAttenuationEnabled(False)
            self.setRippleEnabled(False)
            self.setTransitionWidthEnabled(False)
        elif value == 'Hamming':
            self.setAlphaEnabled(True)
            self.setStopBandAttenuationEnabled(False)
            self.setRippleEnabled(False)
            self.setTransitionWidthEnabled(False)
        elif value == 'Hanning':
            self.setAlphaEnabled(True)
            self.setStopBandAttenuationEnabled(False)
            self.setRippleEnabled(False)
            self.setTransitionWidthEnabled(False)
        elif value == 'Kaiser':
            self.setAlphaEnabled(False)
            self.setStopBandAttenuationEnabled(True)
            self.setRippleEnabled(False)
            self.setTransitionWidthEnabled(False)
        elif value == 'Chebyshev':
            self.setAlphaEnabled(False)
            self.setStopBandAttenuationEnabled(False)
            self.setRippleEnabled(True)
            self.setTransitionWidthEnabled(True)

    def onAccepted(self):
        """Called when the user clicks the OK button
        """
        self.calculate()

        self.edit.addHistory(self.edit.curSelection[1], f'todo: add filter type info here', f'Filter')


    def onRejected(self):
        """Called when the user clicks the Cancel button
        """
        self.close()

    def setResolution(self, resolution):
        """Set the data resolution.
        """
        self.resolution = resolution
        if self.resolution < 0:
            self.resolution = 1
        self.nyquistFreq = 1.0 / (2.0 * self.resolution)
        self.lowCutoffFreq = 0.1 * self.nyquistFreq
        self.highCutoffFreq = 0.5 * self.nyquistFreq
        self.transitionWidth = 0.1 * self.nyquistFreq
        self.displayNyquistFreq()
        self.setWidgets()

    def chebyshevParameters(self):
        """
        """
        self.getValues()
        numPoints, ripple, transitionWidth = chebyshevParameters(self.numPoints, self.ripple, self.transitionWidth)
        self.numPoints = numPoints
        self.ripple = ripple
        self.transitionWidth = transitionWidth

    def updateFilterType(self):
        """Update the user-selectable filter type.
        """
        filterTypeList = [self.lowPass, self.highPass, self.bandPass, self.bandPass]
        self.filterTypeIndex = self.ui.filterTypeComboBox.currentIndex()
        self.generateFilter = filterTypeList[self.filterTypeIndex]

    def updateWindowType(self):
        """Update the user-selectable window type.
        """
        windowTypeList = [self.rectangular, self.triangular, self.hamming, self.hamming, self.kaiser, self.chebyshev]
        self.windowTypeIndex = self.ui.windowTypeComboBox.currentIndex()
        self.generateWindow = windowTypeList[self.windowTypeIndex]

    def displayNyquistFreq(self):
        """Display the Nyquist frequency.
        """
        self.ui.nyquistFreqText.setText("{:.6f}".format(self.nyquistFreq))

    # Filter list (g) generators

    def lowPass(self):
        """Low-pass filter
        """
        g = lowpassResponse(self.numPoints + self.noff, self.highCutoffFreq, self.resolution)
        return g

    def highPass(self):
        """High-pass filter
        """
        g = lowpassResponse(self.numPoints + self.noff, self.lowCutoffFreq, self.resolution)
        return g

    def bandPass(self):
        """Band-pass filter and band-stop filter
        """
        g = bandpassResponse(self.numPoints + self.noff, self.lowCutoffFreq, self.highCutoffFreq, self.resolution)
        return g

    # Coefficients list (w) generators

    def rectangular(self):
        """Rectangular window
        """
        # rectangular set coefficients as 1
        nfw = self.numPoints
        if self.windowType == 'Hanning':
            nfw = self.numPoints + 2
        bounds = int((nfw + 1) / 2)
        w = [1 for i in range(0, bounds)]
        return w

    def triangular(self):
        """Triangular window
        """
        w = triangularWindow(self.numPoints)
        return w

    def hamming(self):
        """Hamming and Hanning window
        """
        beta = 1.0 - self.alpha
        w = hammingWindow(self.numPoints + self.noff, self.alpha, beta)
        return w

    def kaiser(self):
        """Kaiser window
        """
        att = self.attenuation
        beta = 0.0
        if att > 20.96:
            if att < 50.0:
                k = att - 20.96
                beta = 0.58417 * pow(k, 0.4) + 0.07886 * k
            else:
                beta = 0.1102 * (att - 8.7)
        w = kaiserWindow(self.numPoints, beta)
        return w

    def chebyshev(self):
        """Chebyshev window
        """
        w = chebyshevWindow(self.numPoints, self.ripple, self.transitionWidth)
        return w

    # Adjustors after g * w

    def adjustHighPassOrBandStop(self):
        """
        """
        half = self.halfNumPoints
        G = self.gw
        if G is None:
            return
        g = [0] * len(G)
        g[0] = 1 - G[0]
        g[1:half] = [-value for value in G[1:half]]
        g[half:] = [value for value in G[half:]]
        return g

    def adjustLowPass(self):
        """
        """
        half = self.halfNumPoints
        G = self.gw
        b = sum(G[1:half])
        b = 2 * b + G[0]
        g = [0] * len(G)
        g[0:half] = [value / b for value in G[0:half]]
        g[half:-1] = G[half:-1]
        return g

    def calculate(self):
        """Filters the data.
        """
        self.getValues()
        if self.windowType == 'Hanning':
            self.noff = 2
        else:
            self.noff = 0
        self.g = self.generateFilter()
        self.w = self.generateWindow()
        n = len(self.g)
        self.gw = self.g

        if self.windowType == 'Rectangular':
            self.gw = [self.g[i] for i in range(n)]
        else:
            self.gw = [self.g[i] * self.w[i] for i in range(n)]

        if self.filterType == 'High Pass' or self.filterType == 'Band Stop':
            self.gw = self.adjustHighPassOrBandStop()

        if self.filterType == 'Low Pass':
            self.gw = self.adjustLowPass()

        if self.numPoints != self.oldNumPoints:
            pass

        # just filter anything that is plotted for now
        # later could make a separate dstr selection window (using the axis ones doesn't make sense as filters operate on data independently)
        for plotStrs in self.parent.lastPlotStrings:
            for dstr in plotStrs:
                print(f'filtering {dstr}')
                data = self.filterRawData(self.parent.getData(dstr))
                self.parent.DATADICT[dstr].append([self.parent.totalEdits, data, f'{dstr}_fltr'])

    def filterRawData(self, data):
        hnf = self.halfNumPoints
        G = [0] * self.numPoints
        G[0:hnf] = [self.gw[hnf - i - 1] for i in range(hnf)]
        G[hnf:] = [self.gw[i - hnf + 1] for i in range(hnf - 1)]
        return self.applyFilter(G, data)

    def applyFilter(self, G, vin):
        num = len(vin) - self.numPoints
        J = len(G)
        vout = np.empty(num)
        for i in range(num):
            vout[i] = sum(G * vin[i:i+J])
        return vout
