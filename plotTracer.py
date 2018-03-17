
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

        # add spacer to fgrid
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ui.fgrid.addItem(spacer,0,1,1,1)

        self.checkBoxes = []
        self.fcheckBoxes = []
        self.fLabels = 0

        # lastPlotMatrix is set whenever window does a plot
        if self.window.lastPlotMatrix is not None:
            for i,axis in enumerate(self.window.lastPlotMatrix):
                self.addPlot()
                for j,checked in enumerate(axis):
                    self.checkBoxes[i][j].setChecked(checked)

    def plotData(self):
        boolMatrix = []
        for cbAxis in self.checkBoxes:
            boolAxis = []
            for cb in cbAxis:
                boolAxis.append(cb.isChecked())
            boolMatrix.append(boolAxis)
        self.window.lastPlotMatrix = boolMatrix
        self.window.plotData(boolMatrix, False)

    def clearCheckBoxes(self):
        for row in self.checkBoxes:
            for cb in row:
                cb.setChecked(False)

    # callback for each checkbox on changed
    # which box, row and col are provided
    def checkPlotLinks(self, checkBox, r, c):
        print(f'{r} {c}')
        maxRows = int(self.plotCount / 2) # max number of rows required
        return

        if checkBox.isChecked(): # make sure ur only one checked in your column
            for i in range(len(self.fcheckBoxes)):
                if i != r: # skip self
                    self.fcheckBoxes[i][c].setChecked(False)

        targRows = self.getProperLinkRowCount()
        while len(self.fcheckBoxes) < targRows:
            self.addLinkRow()
        while len(self.fcheckBoxes) > targRows:
            self.removeLinkRow()

    def addLinkRow(self): # adds to end
        fgrid = self.ui.fgrid

        row = []
        r = len(self.fcheckBoxes)

        linkGroupLabel = QtWidgets.QLabel()
        linkGroupLabel.setText(f'LinkGroup{r+1}')
        fgrid.addWidget(linkGroupLabel,r+1,0,1,1)

        for i in range(self.plotCount):
            checkBox = QtWidgets.QCheckBox()
            checkBox.setChecked(False)
            checkBox.stateChanged.connect(functools.partial(self.checkPlotLinks, checkBox, r, i))
            row.append(checkBox)
            fgrid.addWidget(checkBox,r+1,i+1,1,1)       

        self.fcheckBoxes.append(row) # adds to end

    def removeLinkRow(self): # removes a most empty row from closest to end (moves the rest down)
        minCount = 1000
        rowIndex = -1
        for r,row in enumerate(self.fcheckBoxes):
            count = 0
            for cb in row:
                if cb.isChecked():
                    count += 1
            if count <= minCount: # <= so later rows are favored
                minCount = count
                rowIndex = r

        if rowIndex == -1:
            print('BAD ROW INDEX')
            return

        fgrid = self.ui.fgrid

        # remove target row of checkboxes
        del self.fcheckBoxes[rowIndex]
        for i in range(1, self.plotCount+1): # skip labels
            item = fgrid.itemAtPosition(rowIndex+1, i)
            fgrid.removeItem(item)
            item.widget().deleteLater()

        # move other rows of checkboxes down by one
        gridRows = len(self.fcheckBoxes)+2
        for r in range(rowIndex+2, gridRows):
            for i in range(1, self.plotCount+1):
                item = fgrid.itemAtPosition(r, i)
                fgrid.removeItem(item)
                fgrid.addItem(item, r-1, i)
        
        # delete last left label
        item = fgrid.itemAtPosition(gridRows-1, 0)
        fgrid.removeItem(item)
        item.widget().deleteLater()
    
    def getProperLinkRowCount(self):
        targ = int(self.plotCount / 2) # max number of rows possibly required
        return targ

        # loop through each checkbox and check for more than one empty row
        freeRows = 0
        for row in self.fcheckBoxes:
            count = 0
            for cb in row:
                if cb.isChecked():
                    count += 1
            if count < 2:
                freeRows += 1

        while freeRows > 1:
            freeRows -= 1
            targ -= 1

        return targ

    # called on add or remove axis
    # resize each row to match with self.plotCount
    # also need to change top labels
    def resizeColumns(self):
        fgrid = self.ui.fgrid

        while self.fLabels < self.plotCount:
            # take out spacer
            spacer = fgrid.itemAtPosition(0,self.fLabels+1)
            fgrid.removeItem(spacer)
            # add new plot label
            pLabel = QtWidgets.QLabel()
            pLabel.setText(f'Plot{self.fLabels+1}')
            pLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            fgrid.addWidget(pLabel,0,self.fLabels+1,1,1)
            # add back spacer
            fgrid.addItem(spacer,0,self.fLabels+2,1,1)
            self.fLabels += 1

        while self.fLabels > self.plotCount:
            # take out spacer
            spacer = fgrid.itemAtPosition(0,self.fLabels+1)
            fgrid.removeItem(spacer)
            # take out plot label
            pLabel = fgrid.itemAtPosition(0,self.fLabels)
            fgrid.removeItem(pLabel)
            pLabel.widget().deleteLater()
            # add back spacer
            fgrid.addItem(spacer,0,self.fLabels,1,1)
            self.fLabels -= 1

        targRows = self.getProperLinkRowCount()

        while len(self.fcheckBoxes) < targRows:
            self.addLinkRow()
        while len(self.fcheckBoxes) > targRows:
            self.removeLinkRow()

        if len(self.fcheckBoxes) == 0:
            return
        
        while len(self.fcheckBoxes[0]) < self.plotCount: # add column to end
            i = len(self.fcheckBoxes[0])
            for r,row in enumerate(self.fcheckBoxes):
                checkBox = QtWidgets.QCheckBox()
                checkBox.setChecked(False)
                checkBox.stateChanged.connect(functools.partial(self.checkPlotLinks, r, i))
                row.append(checkBox)
                fgrid.addWidget(checkBox,r+1,i+1,1,1)                        

        while len(self.fcheckBoxes[0]) > self.plotCount: # remove column from end
            i = len(self.fcheckBoxes[0])            
            for r,row in enumerate(self.fcheckBoxes):
                row.pop() # remove last checkBox from list
                checkBox = fgrid.itemAtPosition(r+1,i)
                fgrid.removeItem(checkBox)
                checkBox.widget().deleteLater()



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

        self.resizeColumns()

    def removePlot(self):
        if self.plotCount == 0:
            print('no plots to delete')
            return
        self.plotCount-=1

        rowLen = len(DATASTRINGS)+1 #+1 for plot label
        self.checkBoxes = self.checkBoxes[:-1]

        for i in range(rowLen):
            child = self.ui.grid.takeAt(self.ui.grid.count()-1) # take items off back
            if child.widget() is not None:
                child.widget().deleteLater()

        self.ui.grid.invalidate()
        self.resizeColumns()        

