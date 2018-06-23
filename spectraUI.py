
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from pyqtgraphExtensions import GridGraphicsLayout, LinearGraphicsLayout, LogAxis, BLabelItem
from MagPy4UI import TimeEdit

class SpectraUI(object):

    def buildSpectraView(self):
        gview = pg.GraphicsView()
        gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        gmain = LinearGraphicsLayout() # made this based off pg.GraphicsLayout
        #apparently default is 11, tried getting the margins and they all were zero seems bugged according to pyqtgraph
        gmain.setContentsMargins(11,0,11,0) # left top right bottom
        gview.setCentralItem(gmain)
        grid = GridGraphicsLayout()
        grid.setContentsMargins(0,0,0,0)
        labelLayout = GridGraphicsLayout()
        labelLayout.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Minimum))
        gmain.addItem(grid)
        gmain.addItem(labelLayout)
        return gview, grid, labelLayout

    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Spectra')
        Frame.resize(1000,700)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.gview, self.grid, self.labelLayout = self.buildSpectraView()

        self.tabs.addTab(self.gview, 'Spectra')

        #self.cohView, self.cohGrid, self.cohLabelLayout = self.buildSpectraView()

        #self.tabs.addTab(self.cohView, 'Coherence')

        #self.phaseView, self.phaseGrid, self.phaseLabelLayout = self.buildSpectraView()

        #self.tabs.addTab(self.phaseView, 'Phase')

        #layout.addWidget(self.gview)

        # bandwidth label and spinbox
        bottomLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtGui.QLabel("Average Bandwidth")
        self.bandWidthSpinBox = QtGui.QSpinBox()
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setProperty("value", 3)

        self.separateTracesCheckBox = QtGui.QCheckBox()
        self.separateTracesCheckBox.setChecked(True)
        separateTraces = QtGui.QLabel("Separate Traces")

        self.aspectLockedCheckBox = QtGui.QCheckBox()
        self.aspectLockedCheckBox.setChecked(True)
        aspectLockedLabel = QtGui.QLabel("Lock Aspect Ratio")


        optFrame = QtWidgets.QGroupBox()

        optLayout = QtWidgets.QGridLayout(optFrame)
        optLayout.addWidget(bandWidthLabel, 0, 0, 1, 1)
        optLayout.addWidget(separateTraces, 1, 0, 1, 1)
        optLayout.addWidget(aspectLockedLabel, 2, 0, 1, 1)
        optLayout.addWidget(self.bandWidthSpinBox, 0, 1, 1, 1)
        optLayout.addWidget(self.separateTracesCheckBox, 1, 1, 1, 1)
        optLayout.addWidget(self.aspectLockedCheckBox, 2, 1, 1, 1)

        bottomLayout.addWidget(optFrame)

        timeFrame = QtWidgets.QGroupBox()
        timeLayout = QtWidgets.QVBoxLayout(timeFrame)

        # setup datetime edits
        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 10 if window.OS == 'windows' else 14))
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        timeLayout.addWidget(self.timeEdit.start)
        timeLayout.addWidget(self.timeEdit.end)

        bottomLayout.addWidget(timeFrame)

        #cohPair1 = QtWidgets.QComboBox()
        #bottomLayout.addWidget(cohPair1)
        #cohPair2 = QtWidgets.QComboBox()
        #bottomLayout.addWidget(cohPair2)
        #for dstrs in window.lastPlotStrings:
        #    for dstr in dstrs:
        #        cohPair1.addItem(dstr)
        #        cohPair2.addItem(dstr)

        bottomLayout.addStretch()

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        bottomLayout.addWidget(self.updateButton)


        layout.addLayout(bottomLayout)
        


class SpectraViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    # overriding part of this function to get resizing to work correctly with
    # manual y range override and fixed aspect ratio settings
    def updateViewRange(self, forceX=False, forceY=False):
        tr = self.targetRect()
        bounds = self.rect()
        aspect = self.state['aspectLocked']
        if aspect is not False and 0 not in [aspect, tr.height(), bounds.height(), bounds.width()]:
            targetRatio = tr.width()/tr.height() if tr.height() != 0 else 1
            viewRatio = (bounds.width() / bounds.height() if bounds.height() != 0 else 1) / aspect
            viewRatio = 1 if viewRatio == 0 else viewRatio
            if viewRatio > targetRatio:
                pg.ViewBox.updateViewRange(self,False,True) 
                return
        pg.ViewBox.updateViewRange(self,forceX,forceY) #default