
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from MagPy4UI import UI_PlotTracer
import functools

from dataLayout import *

class PlotTracer(QtWidgets.QFrame, UI_PlotTracer):
    def __init__(self, window, parent=None):
        super(PlotTracer, self).__init__(parent)

        self.window = window
        self.ui = UI_PlotTracer()
        self.ui.setupUI(self)
        self.plotCount = 0

        #self.ui.drawStyleCombo.currentIndexChanged.connect(self.setLineStyle)
        self.ui.clearButton.clicked.connect(self.clearCheckBoxes)
        self.ui.addPlotButton.clicked.connect(self.addPlot)
        self.ui.removePlotButton.clicked.connect(self.removePlot)
        self.ui.plotButton.clicked.connect(self.plotData)

        self.addLabels()

        self.checkBoxes = []
        self.fcheckBoxes = []
        self.fLabels = 0

        # set up previous saved plot and link checkboxes
        if self.window.lastPlotMatrix is not None:
            for i,axis in enumerate(self.window.lastPlotMatrix):
                self.addPlot()
                for j,b in enumerate(axis):
                    self.checkBoxes[i][j].setChecked(b)
        if self.window.lastLinkMatrix is not None:
            numRows = len(self.window.lastLinkMatrix)
            self.rebuildPlotLinks(numRows)
            for i,axis in enumerate(self.window.lastLinkMatrix):
                for j,b in enumerate(axis):
                    self.fcheckBoxes[i][j].setChecked(b)

    # returns bool matrix from checkbox matrix
    def checksToBools(self, cbMatrix, skipEmpty = False):
        boolMatrix = []
        for cbAxis in cbMatrix:
            boolAxis = []
            nonEmpty = False
            for cb in cbAxis:
                b = cb.isChecked()
                nonEmpty = nonEmpty or b
                boolAxis.append(b)
            if nonEmpty or not skipEmpty:
                boolMatrix.append(boolAxis)
        return boolMatrix

    def plotData(self):
        boolMatrix = self.checksToBools(self.checkBoxes)
        self.window.lastPlotMatrix = boolMatrix
        self.window.plotData(boolMatrix, self.checksToBools(self.fcheckBoxes))

    def clearCheckBoxes(self):
        for row in self.checkBoxes:
            for cb in row:
                cb.setChecked(False)

    # callback for each checkbox on changed
    # which box, row and col are provided
    def checkPlotLinks(self, checkBox, r, c):
        #print(f'{r} {c}')
        if checkBox.isChecked(): # make sure ur only one checked in your column
            i = 0 # need to do it like this because setChecked callbacks can cause links to be rebuild mid iteration
            while i < len(self.fcheckBoxes):
                if i != r: # skip self
                    #print(f'{i} {c} {len(self.fcheckBoxes)} {len(self.fcheckBoxes[0])}')
                    self.fcheckBoxes[i][c].setChecked(False)
                i += 1

        if self.targRows != self.getProperLinkRowCount():
            self.rebuildPlotLinks()

    def getProperLinkRowCount(self):
        targ = int(self.plotCount / 2) # max number of rows possibly required
        used = 1   # what this part does is make sure theres at least 1 row with < 2
        for row in self.fcheckBoxes:
            count = 0
            for cb in row:
                count += 1 if cb.isChecked() else 0
            if count >= 2:
                used += 1
        #print(f'{used if used < targ else targ}')
        return used if used < targ else targ

    # rebuild whole link table (tried adding and removing widgets and got pretty close but this was way easier in hindsight)
    def rebuildPlotLinks(self, targRows = None):
        fgrid = self.ui.fgrid

        if targRows is None:
            self.targRows = self.getProperLinkRowCount()
        else:
            self.targRows = targRows

        cbools = self.checksToBools(self.fcheckBoxes, True)
        self.fcheckBoxes = [] # clear list after

        # clear whole layout
        while fgrid.count():
            item = fgrid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        # add top plot labels
        for i in range(self.plotCount):
            pLabel = QtWidgets.QLabel()
            pLabel.setText(f'Plot{i+1}')
            pLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            fgrid.addWidget(pLabel,0,i+1,1,1)
        # add spacer
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        fgrid.addItem(spacer,0,self.plotCount+1,1,1)

        # add checkBoxes
        for r in range(self.targRows):
            linkGroupLabel = QtWidgets.QLabel()
            linkGroupLabel.setText(f'Group{r+1}')
            fgrid.addWidget(linkGroupLabel,r+1,0,1,1)
            row = []
            for i in range(self.plotCount):
                checkBox = QtWidgets.QCheckBox()
                if r < len(cbools) and len(cbools) > 0 and i < len(cbools[0]):
                    checkBox.setChecked(cbools[r][i])
                checkBox.stateChanged.connect(functools.partial(self.checkPlotLinks, checkBox, r, i))
                row.append(checkBox)
                fgrid.addWidget(checkBox,r+1,i+1,1,1)      
            self.fcheckBoxes.append(row)

    def addLabels(self):
        self.labels = []
        for i,dstr in enumerate(DATASTRINGS):
            label = QtWidgets.QLabel()
            label.setText(dstr)
            label.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            self.ui.grid.addWidget(label,0,i+1,1,1)

    def addPlot(self):
        self.plotCount += 1
        plotLabel = QtWidgets.QLabel()
        plotLabel.setText(f'Plot{self.plotCount}')
        checkBoxes = []
        a = self.plotCount + 1 # first axis is labels so +1
        self.ui.grid.addWidget(plotLabel,a,0,1,1)
        for i,dstr in enumerate(DATASTRINGS):
            checkBox = QtWidgets.QCheckBox()
            checkBoxes.append(checkBox)
            self.ui.grid.addWidget(checkBox,a,i+1,1,1)
        self.checkBoxes.append(checkBoxes)

        self.rebuildPlotLinks()

    def removePlot(self):
        if self.plotCount == 0:
            print('no plots to delete')
            return
        self.plotCount-=1

        rowLen = len(DATASTRINGS)+1 # +1 for plot label
        self.checkBoxes = self.checkBoxes[:-1]

        for i in range(rowLen):
            child = self.ui.grid.takeAt(self.ui.grid.count()-1) # take items off back
            if child.widget() is not None:
                child.widget().deleteLater()

        self.ui.grid.invalidate() #otherwise gridframe doesnt shrink back down
        self.rebuildPlotLinks()        

