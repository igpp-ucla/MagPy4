

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(500,500)

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

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont=QtGui.QFont()
        self.tickFont.setPixelSize(14)

    def tickStrings(self, values, scale, spacing):
        return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        
        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity
        
        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        ## Draw all text
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            qst = QtGui.QStaticText(f'10<sup>{text}</sup>')
            qst.setTextFormat(QtCore.Qt.RichText)
            p.drawStaticText(rect.left(), rect.top(), qst)

            #p.drawText(rect, flags, text)
            #p.drawRect(rect)

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

            leftAxis = LogAxis(orientation='left')
            bottomAxis = LogAxis(orientation='bottom')
            pi = pg.PlotItem(axisItems={'bottom':bottomAxis, 'left':leftAxis})
            pi.setLogMode(True, True) #todo redo ticks to be of format 10^x
            print(f'freqLen {len(freq)} powLen {len(power)}')

            pi.plot(freq, power, pen=self.window.pens[0])

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

            power[n] = sum(fsqr[kO * 2 - 1:kE * 2 - 1]) / bw * C

        return power

    def calculateCoherence(self):
        pass


        