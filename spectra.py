

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from  LinearGraphicsLayout import LinearGraphicsLayout

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(500,500)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout.addWidget(self.glw)

        #self.gview = pg.GraphicsView()
        #self.glw = LinearGraphicsLayout()
        #self.gview.setCentralItem(self.glw)
        #layout.addWidget(self.gview)

        # bandwidth label and spinbox
        bandWidthLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtGui.QLabel("Average Bandwidth")
        self.bandWidthSpinBox = QtGui.QSpinBox()
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setProperty("value", 3)
        self.oneTraceCheckBox = QtGui.QCheckBox()
        self.oneTraceCheckBox.setChecked(True)
        oneTraceLabel = QtGui.QLabel("One Trace Per Plot")

        bandWidthLayout.addWidget(bandWidthLabel)
        bandWidthLayout.addWidget(self.bandWidthSpinBox)
        bandWidthLayout.addWidget(oneTraceLabel)
        bandWidthLayout.addWidget(self.oneTraceCheckBox)
        bandWidthLayout.addStretch()

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        layout.addLayout(bandWidthLayout)
        layout.addWidget(self.updateButton)

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont=QtGui.QFont()
        self.tickFont.setPixelSize(14)
        self.style['maxTextLevel'] = 1 # never have any subtick labels
        self.style['textFillLimits'] = [(0,1.1)] # try to always draw labels
        #self.style['tickLength'] = -10
        #todo: override AxisItem generateDrawSpecs and custom set tick length

    def tickStrings(self, values, scale, spacing):
        return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below

    def tickSpacing(self, minVal, maxVal, size):
        #levels = pg.AxisItem.tickSpacing(self,minVal,maxVal,size)
        #if len(levels) > 2:
        #    levels.pop()
        #print(levels)  
        levels = [(10.0,0),(1.0,0),(0.5,0)]
        return levels

    # overriden from source to be able to have superscript text
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

class SpectraInfiniteLine(pg.InfiniteLine):
    def __init__(self, window, index, *args, **kwds):
        pg.InfiniteLine.__init__(self, *args, **kwds)
        self.window = window
        self.index = index

    def mouseDragEvent(self, ev):
        pg.InfiniteLine.mouseDragEvent(self, ev)
        #update all other infinite spectra lines on left or right depending
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            x = self.getXPos()
            self.window.updateSpectra(self.index,x)

class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self)
        
        self.ui.updateButton.clicked.connect(self.updateSpectra)

        self.updateSpectra()

    def closeEvent(self, event):
        # hide spectra lines in window
        self.window.spectraStep = 0
        for pi in self.window.plotItems:
            for line in pi.getViewBox().spectLines:
                line.hide()

    def updateSpectra(self):
        dataStrings = self.window.getSpectraPlots()
        indices = self.window.getSpectraRangeIndices()
        self.N = indices[1] - indices[0]
        #print(self.N)
        freq = self.calculateFreqList()
        if freq is None:
            return

        self.ui.glw.clear()
        self.ui.glw.ci.currentRow = 0 # clear doesnt get rid of layout formatting correctly it seems
        self.ui.glw.ci.currentCol = 0
        oneTracePerPlot = self.ui.oneTraceCheckBox.isChecked()
        #curLayout = self.ui.glw.addLayout()
        numberPlots = 0
        for strList in dataStrings:
            for i,dstr in enumerate(strList):
                if i == 0 or oneTracePerPlot:
                    pi = pg.PlotItem(axisItems={'bottom':LogAxis(orientation='bottom'), 'left':LogAxis(orientation='left')})
                    titleString = ''
                    pi.setLogMode(True, True)
                    numberPlots += 1

                p = self.window.pens[min(i,len(self.window.pens) - 1)]
                data = self.window.DATADICT[dstr][indices[0]:indices[1]]
                fft = fftpack.rfft(data.tolist())
                power = self.calculatePower(fft)
                titleString = f"{titleString} <span style='color:{p.color().name()};'>{dstr}</span>"
                #f"<span style='color:{p.color().name()};'>{dstr}</span>\n"

                #print(f'freqLen {len(freq)} powLen {len(power)}')

                pi.plot(freq, power, pen=p)

                if i == len(strList)-1 or oneTracePerPlot:
                    pi.setLabels(title=titleString, left='Power', bottom='Frequency')
                    self.ui.glw.addItem(pi)
                    #curLayout.addItem(pi)
                    if numberPlots % 4 == 0:
                        self.ui.glw.nextRow()
                        #curLayout = self.ui.glw.addLayout()
                        pass

        #labelLayout = LinearGraphicsLayout(orientation = QtCore.Qt.Horizontal)
        #self.ui.glw.addItem(labelLayout)

        self.ui.glw.nextRow()
        # draw some text info like time range and file and stuff
        # add Y axis label based on traces

        #lw = self.ui.glw.ci.layout
        #lw.setColumnStretchFactor(1,100)

        li = pg.LabelItem()
        #li.item.setHtml(f"<span style='font-size:{fontSize}pt; white-space:pre;'>{axisString}</span>")
        li.item.setHtml(f'hello this is a test wait wat no i dont want to test<br>here is a date<br>102-300:100:100>>>10000')
        #filename
        #time range, resolution
        #number freq samples or something
        self.ui.glw.addItem(li)
        #labelLayout.addItem(li)

        ## end of def

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


        