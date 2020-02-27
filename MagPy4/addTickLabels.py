from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from .pyqtgraphExtensions import LinkedAxis, DateAxis
import functools
import bisect
import numpy as np

class AddTickLabelsUI(object):
    def setupUI(self, Frame, window, dstrList, prevDstrs):
        Frame.setWindowTitle('Additional Tick Labels')
        Frame.resize(100, 100)
        layout = QtWidgets.QGridLayout(Frame)

        # Initialize checkbox elements for all dstrs not plotted or with
        # tick labels set
        self.chkboxes = []
        maxWidth = 4 # Limit the number of checkboxes per row
        if len(dstrList) > 30:
            maxWidth = 9
        dstrNum, rowNum = 0, 0
        for dstr in dstrList:
            chkbx = QtWidgets.QCheckBox(dstr)
            if dstrNum != 0 and dstrNum % maxWidth == 0:
                rowNum += 1
            layout.addWidget(chkbx, rowNum, dstrNum % maxWidth, 1, 1)
            self.chkboxes.append(chkbx)
            dstrNum += 1
        
        # Add in checkboxes for all previously set tick labels
        for dstr in prevDstrs:
            if dstr in dstrList:
                # Set previously added labels' boxes as checked
                index = dstrList.index(dstr)
                self.chkboxes[index].setChecked(True)
            else: # In case also plotted, add UI element so it can be removed if needed
                numDstrs = len(self.chkboxes)
                rowNum = int(numDstrs/maxWidth) # Calculate row and column nums
                colNum = numDstrs % maxWidth
                chkbx = QtWidgets.QCheckBox(dstr)
                layout.addWidget(chkbx, rowNum, colNum, 1, 1)
                self.chkboxes.append(chkbx)
                chkbx.setChecked(True)

class AddTickLabels(QtGui.QFrame, AddTickLabelsUI):
    def __init__(self, window, pltGrd, parent=None):
        super(AddTickLabels, self).__init__(parent)
        self.ui = AddTickLabelsUI()
        self.window = window
        self.pltGrd = pltGrd

        # Get list of all dstrs not currently plotted
        allDstrs = window.DATASTRINGS[:]
        for dstrList in window.lastPlotStrings:
            for dstr, en in dstrList:
                if dstr in allDstrs: # Remove currently plotted dstrs
                    allDstrs.remove(dstr)
        
        # Get list of all dstrs that currently have extra tick labels set
        prevDstrs = []
        if self.pltGrd.labelSetGrd:
            prevDstrs = [lbl.dstr for lbl in self.pltGrd.labelSetGrd.labelSets]
        
        # Set up UI based on dstrs not plotted and check/add all current labelsets
        self.ui.setupUI(self, window, allDstrs, prevDstrs)

        # Connect every checkbox to function
        for chkbx in self.ui.chkboxes:
            chkbx.clicked.connect(functools.partial(self.addLabelSet, chkbx))

    def addLabelSet(self, chkbx):
        # Add to pltGrd if chkbx is checked, remove it if unchecked
        dstr = chkbx.text()
        if chkbx.isChecked():
            self.pltGrd.addLabelSet(dstr)
        else:
            self.pltGrd.removeLabelSet(dstr)

class invisAxis(DateAxis):
    # Axis item that hides all lines and only paints text items at ticks
    def __init__(self, window, dstr, orientation):
        self.window = window
        self.dstr = dstr
        DateAxis.__init__(self, window.epoch, orientation)

    def tickStrings(self, values, scale, spacing):
        indices = self.getDataIndices(self.dstr, values)
        strings = []
        data = self.window.getData(self.dstr, self.window.currentEdit)
        for index in indices:
            val = data[index]
            txt = str(np.round(val, decimals=4))
            strings.append(txt)

        return strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        
        ## Draw all text
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            p.drawText(rect, flags, text)

    def getDataIndices(self, dstr, tickValues):
        # Uses bisect search to get all the data indices corresp. to each tick
        dataIndices = []
        times = self.window.getTimes(dstr, 0)[0]
        for tv in tickValues:
            curIndex = bisect.bisect(times, tv + self.window.tickOffset)
            dataIndices.append(curIndex)
        return dataIndices

class LabelSet(pg.PlotItem):
    # PlotItem subclass used to draw additional tick labels to match main plot's ticks
    # This was simpler than trying to adjust resize events with only an axisItem
    def __init__(self, window, dstr, parent=None, name=None, labels=None, title=None, viewBox=None, axisItems=None, enableMenu=True, **kargs):
        self.window = window
        self.dstr = dstr

        # Initialize axisItems to be used to show labels and for resizing
        topAxis = invisAxis(window, dstr, 'top')
        topAxis.tickOffset = window.tickOffset
        rightAxis = LinkedAxis(orientation='right')
        axisDict = {'top': topAxis, 'right': rightAxis}
        pg.PlotItem.__init__(self, parent, name, labels, title, viewBox, enableMenu=False, axisItems=axisDict, **kargs)

        # Reduce the plotItem to just the invisAxis item, visually.
        self.getAxis('left').setHeight(0)
        self.getAxis('right').setHeight(0)
        self.layout.removeItem(self.vb) # Remove empty viewbox
        self.hideAxis('bottom')
        self.showAxis('top')

        # Disable interactive elements of plotItem
        self.hideButtons()
        self.setMouseEnabled(False)
    
    def setCstmTickSpacing(self, diff):
        self.getAxis('top').setCstmTickSpacing(diff)

class LabelSetGrid(pg.GraphicsLayout):
    # GraphicsLayout used to easily synchronize all resize events and tick
    # value updates across label sets, as well as adding new label sets
    def __init__(self, window, *args, **kwargs):
        self.window = window
        self.labelSets = []
        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.layout.setVerticalSpacing(1)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setColumnAlignment(0, QtCore.Qt.AlignBottom)
    
    def addLabelSet(self, dstr):
        # Creates a new label set from the given dstr and adds it to grid
        labelSet = LabelSet(self.window, dstr)
        self.addItem(labelSet, len(self.labelSets), 0, 1, 1)
        self.layout.setRowAlignment(len(self.labelSets), QtCore.Qt.AlignCenter)
        self.labelSets.append(labelSet)
        startTime = self.window.tO-self.window.tickOffset
        endTime = self.window.tE-self.window.tickOffset
        labelSet.setXRange(startTime, endTime, 0)

        # Set tick spacing if set for main grid's time axis
        if self.window.plotItems != []:
            btmPlt = self.window.plotItems[-1]
            diff = btmPlt.getAxis('bottom').tickDiff
            if diff is not None:
                labelSet.setCstmTickSpacing(diff)
    
    def adjustWidths(self, width):
        # Used by pltGrd to sync labelSet widths w/ plot widths
        for lblSt in self.labelSets:
            lblSt.getAxis('left').setWidth(width)

    def setXRange(self, x0, x1, padding=0):
        # Update tick locs/labels for all label sets
        for lblSt in self.labelSets:
            lblSt.setXRange(x0, x1, padding)

    def setCstmTickSpacing(self, diff):
        for lblSet in self.labelSets:
            lblSet.setCstmTickSpacing(diff)