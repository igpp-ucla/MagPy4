from pyqtgraphExtensions import LinkedRegion
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from dataDisplay import UTCQDate
from FF_File import FFTIME
import functools

class GeneralSelect(object):
    '''
        Object that manages region selection in plot grid, including:
            - Multi-selection and region adjustments
            - Sub-region hiding
            - Extracting selected plot info
            - Calling start/update functions for the tool
    '''

    def __init__(self, window, mode, name='Select', color=None, timeEdit=None, 
        func=None, updtFunc=None, closeFunc=None, maxSteps=1):
        self.window = window
        self.mode = mode
        self.name = name
        self.color = color
        self.timeEdit = timeEdit
        self.func = func # Function called once region fully selected
        self.updtFunc = updtFunc # Function called if lines changes
        self.closeFunc = closeFunc # Function called when all regions hidden
        self.maxSteps = maxSteps # If negative, multi-select mode

        self.allModes = ['Multi','Single','Adjusting','Line']
        '''
            Multi - multiple regions can be selected, not used for now
            Single - single selected region
            Adjusting - starts as a single line and can expand to one or more
                        full regions (limited by maxSteps)
            Line - Single line 'region'
            maxSteps parameter determines how many regions can be created
        '''
        self.regions = []

    def setSubRegionsVisible(self, pltNum, visible=True):
        for region in self.regions:
            region.setVisible(visible, pltNum)

    def stepsLeft(self, ctrlPressed=False):
        if self.maxSteps < 0 and ctrlPressed:
            return True
        elif len(self.regions) < self.maxSteps:
            return True
        else:
            return False

    def leftClick(self, x, pltNum, ctrlPressed=False):
        # Show sub-regions in this plot if applicable
        if not ctrlPressed:
            self.setSubRegionsVisible(pltNum)

        # First region
        if self.regions == []:
            self.addRegion(x)
            return

        # Extend or add new regions depending on settings
        lastRegion = self.regions[-1]
        if lastRegion.fixedLine:
            self.extendRegion(x, lastRegion)
        elif self.stepsLeft(ctrlPressed):
            self.addRegion(x)
        elif self.stepsLeft():
            self.addRegion(x)

        if self.name == 'Stats':
            self.updtFunc()

    def rightClick(self, pltNum):
        # Hide sub regions in this plot
        self.setSubRegionsVisible(pltNum, visible=False)

        # Checks if any region is visible in this plot
        anyVisible = False
        for region in self.regions:
            for pltNum in range(len(region.regionItems)):
                if region.isVisible(pltNum):
                    anyVisible = True
                    break

        if not anyVisible:
            # Call tool-closing-function
            if self.closeFunc:
                self.closeFunc()
            # Remove actual regions from plots
            self.closeAllRegions()
        else:
            if self.name == 'Stats':
                self.updtFunc()

    def closeAllRegions(self):
        # Removes all regions from plot items
        for region in self.regions:
            region.removeRegionItems()

    def addRegion(self, x):
        # Adds a new region/line to all plots and connects it to the timeEdit
        win = self.window
        plts = win.plotItems
        linkRegion = LinkedRegion(win, plts, (x, x), mode=self.name, color=self.color,
            updateFunc=self.updtFunc, linkedTE=self.timeEdit)
        linkRegion.fixedLine = True
        self.connectLinesToTimeEdit(self.timeEdit, linkRegion, self.mode == 'Line')
        self.regions.append(linkRegion)

        if self.mode == 'Line' or self.mode == 'Adjusting':
            if self.func is not None:
                self.openFunc()
            if self.updtFunc:
                self.updtFunc()

    def connectLinesToTimeEdit(self, timeEdit, region, single=False):
        # Makes sure changing time edit updates lines
        if timeEdit == None:
            return
        elif single:
            timeEdit.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
            return
        if self.name == 'Stats' and timeEdit.linesConnected:
            # Disconnect from any previously connected regions (only in Stats mode)
            timeEdit.start.dateTimeChanged.disconnect()
            timeEdit.end.dateTimeChanged.disconnect()
        # Connect timeEdit to currently being moved / recently added region
        timeEdit.start.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
        timeEdit.end.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
        timeEdit.linesConnected = True

    def updateLinesByTimeEdit(self, timeEdit, region, single=False):
        x0, x1 = region.getRegion()
        t0, t1 = self.window.getTimeTicksFromTimeEdit(timeEdit)
        if self.mode == 'Line':
            t0 = self.window.getTimeFromTick(i0)
            self.updateLinesPos(region, t0, t0)
            return
        self.updateLinesPos(region, t0 if x0 < x1 else t1, t1 if x0 < x1 else t0)

    def updateLinesPos(self, region, t0, t1):
        region.setRegion((t0-self.window.tickOffset, t1-self.window.tickOffset))
        if self.updtFunc:
            self.updtFunc()

    def openFunc(self):
        if self.func:
            QtCore.QTimer.singleShot(100, self.func)

    def extendRegion(self, x, region):
        # Extends previously added region (that hasn't been expanded yet)
        if self.mode == 'Line' or self.regions == []:
            return
        x0, x1 = region.getRegion()
        region.setRegion((x0-self.window.tickOffset, x))
        region.fixedLine = False

        region.updateTimeEditByLines(self.timeEdit)

        # Calls open/update functions now that full region is selected
        if self.func is not None:
            self.openFunc()

        if self.updtFunc:
            self.updtFunc()

class SelectableViewBox(pg.ViewBox):
    # Viewbox that calls its window's GeneralSelect object in response to clicks
    def __init__(self, window, plotIndex, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.window = window
        self.plotIndex = plotIndex

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)

        if self.window.currSelect:
            self.window.currSelect.leftClick(x, self.plotIndex, ctrlPressed)
        else:
            self.defaultLeftClick(x, ctrlPressed)

    def defaultLeftClick(self, x, ctrlPressed=False):
        return

    # check if either of lines are visible for this viewbox
    def anyLinesVisible(self):
        isVisible = False
        for region in self.window.regions:
            if region.isVisible(self.plotIndex):
                isVisible = True
        return isVisible

    # sets the lines of this viewbox visible
    def setMyLinesVisible(self, isVisible):
        for region in self.window.regions:
            region.setVisible(isVisible, self.plotIndex)

    def onRightClick(self, ev):
        if self.window.currSelect:
            self.window.currSelect.rightClick(self.plotIndex)
        else:
            pg.ViewBox.mouseClickEvent(self, ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will do same as right click
                #self.onRightClick(ev)
                pass # this seems to confuse people so disabling for now
            else:
               self.onLeftClick(ev)

        else: # asume right click i guess, not sure about middle mouse button click
            self.onRightClick(ev)

        ev.accept()

    # mouse drags for now just get turned into clicks, like on release basically, feels nicer
    # technically only need to do this for spectra mode but not being used otherwise so whatever
    def mouseDragEvent(self, ev, axis=None):
        if ev.isFinish(): # on release
            if ev.button() == QtCore.Qt.LeftButton:
                self.onLeftClick(ev)
            elif ev.button() == QtCore.Qt.RightButton:
                self.onRightClick(ev)
        ev.accept()
        #    pg.ViewBox.mouseDragEvent(self, ev)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()