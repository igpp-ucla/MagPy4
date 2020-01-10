from pyqtgraphExtensions import LinkedRegion
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QSizePolicy
from MagPy4UI import TimeEdit
import pyqtgraph as pg
from dataDisplay import UTCQDate
from FF_File import FFTIME
import functools
import re
import time
import numpy as np
from datetime import datetime
from layoutTools import BaseLayout, TableWidget

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

        self.allModes = ['Multi','Single','Adjusting','Line', 'Multi-Strict']
        '''
            Multi - multiple regions can be selected, not used for now
            Single - single selected region
            Adjusting - starts as a single line and can expand to one or more
                        full regions (limited by maxSteps)
            Line - Single line 'region'
            Multi-Strict - Line by default, other regions must use ctrl
            maxSteps parameter determines how many regions can be created
        '''
        self.regions = []
        self.lblPos = 'top'
        self.movable = True # Whether user can adjust the regions
        self.hidden = False

        # Additional functions to call once user has started selection and
        # after a region is fully selected
        self.stepFunc = None
        self.fullStepFunc = None

    def setStepTrigger(self, func):
        self.stepFunc = func

    def stepTrigger(self):
        if self.stepFunc:
            self.stepFunc()

    def setFullSelectionTrigger(self, func):
        self.fullStepFunc = func
    
    def fullStepTrigger(self):
        if self.fullStepFunc:
            self.fullStepFunc()

    def setLabelPos(self, pos):
        self.lblPos = pos

    def autoSetRegion(self, a, b):
        self.leftClick(a, 0)
        self.leftClick(b, 0)

    def setSubRegionsVisible(self, pltNum, visible=True):
        for region in self.regions:
            region.setVisible(visible, pltNum)
    
    def setAllRegionsVisible(self, val=True):
        for region in self.regions:
            region.setAllRegionsVisible(val)
        self.hidden = not val

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
            self.stepTrigger()
            self.addRegion(x)
            return

        # Extend or add new regions depending on settings
        lastRegion = self.regions[-1]
        if lastRegion.fixedLine:
            self.extendRegion(x, lastRegion, ctrlPressed)
        elif self.stepsLeft(ctrlPressed):
            self.addRegion(x)
        elif self.stepsLeft():
            self.addRegion(x)

        if self.name == 'Stats':
            self.updtFunc()

    def isLine(self):
        if len(regions) < 1:
            return False
        lastRegion = self.regions[-1]
        if lastRegion.fixedLine:
            return True
        return False

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
        linkRegion = self.addLinkedItem(x)
        linkRegion.fixedLine = True
        linkRegion.setMovable(self.movable)
        linkRegion.setAllRegionsVisible(not self.hidden)

        if self.timeEdit:
            self.connectLinesToTimeEdit(self.timeEdit, linkRegion, self.mode == 'Line')
        self.regions.append(linkRegion)

        if self.mode == 'Line' or self.mode == 'Adjusting':
            if self.func is not None:
                self.openFunc()
            if self.updtFunc:
                self.updtFunc()
        
        return linkRegion

    def addLinkedItem(self, x):
        # Initializes the linked region object
        plts = self.window.plotItems
        linkRegion = LinkedRegion(self.window, plts, (x, x), mode=self.name, 
            color=self.color, updateFunc=self.updtFunc, linkedTE=self.timeEdit, 
            lblPos=self.lblPos)
        return linkRegion

    def connectLinesToTimeEdit(self, timeEdit, region, single=False):
        # Makes sure changing time edit updates lines
        if timeEdit == None:
            return
        elif single:
            timeEdit.start.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
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
            self.fullStepTrigger()
            QtCore.QTimer.singleShot(100, self.func)

    def extendRegion(self, x, region, ctrlPressed):
        # Extends previously added region (that hasn't been expanded yet)
        if (self.mode == 'Line' and self.maxSteps >= 0) or self.regions == []:
            return
        x0, x1 = region.getRegion()
        region.setRegion((x0-self.window.tickOffset, x))
        region.fixedLine = False

        if self.timeEdit:
            region.updateTimeEditByLines(self.timeEdit)

        # Calls open/update functions now that full region is selected
        if self.func is not None:
            self.openFunc()

        if self.updtFunc:
            self.updtFunc()

    def setLabelText(self, txt, regionNum=None):
        # Set label text for all regions if none is specified
        if regionNum is None:
            for region in self.regions:
                region.setLabelText(txt)
        else:
            self.regions[regionNum].setLabelText(txt)
    
    def addFullRegion(self, t0, t1):
        region = self.addRegion(t0-self.window.tickOffset)
        self.extendRegion(t1-self.window.tickOffset, region, True)

    def removeRegion(self, regionNum):
        # Remove the specified linked region
        if regionNum >= len(self.regions):
            return
        region = self.regions.pop(regionNum)
        region.removeRegionItems()

    def setMovable(self, val=True, regNum=None):
        if regNum is None:
            for region in self.regions:
                region.setMovable(val)
        else:
            self.regions[regNum].setMovable(val)
        self.movable = val

class SelectableViewBox(pg.ViewBox):
    # Viewbox that calls its window's GeneralSelect object in response to clicks
    def __init__(self, window, plotIndex, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.window = window
        self.plotIndex = plotIndex
        self.setMouseEnabled(x=False, y=False)

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)

        if self.window.currSelect:
            self.window.currSelect.leftClick(x, self.plotIndex, ctrlPressed)
        else:
            self.window.openTraceStats()
            self.window.currSelect = GeneralSelect(self.window, 'Adjusting', 
                'Stats', None, self.window.traceStats.ui.timeEdit,
                None, self.window.updateTraceStats,
                closeFunc=self.window.closeTraceStats, maxSteps=-1)
            self.window.traceStats.ui.timeEdit.linesConnected = False
            self.window.currSelect.leftClick(x, self.plotIndex, ctrlPressed)

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

class FixedSelectionUI():
    def setupUI(self, Frame, window):
        Frame.resize(200, 50)
        Frame.setWindowTitle('Saved Region')
        Frame.move(1200, 0)

        # UI elements
        instr = 'Selected Time Range:'
        lbl = QtWidgets.QLabel(instr)
        self.timeEdit = TimeEdit(QtGui.QFont())

        # Layout setup
        layout = QtWidgets.QGridLayout(Frame)
        layout.addWidget(lbl, 0, 0, 1, 2)
        layout.addWidget(self.timeEdit.start, 1, 0, 1, 1)
        layout.addWidget(self.timeEdit.end, 1, 1, 1, 1)

class FixedSelection(QtWidgets.QFrame, FixedSelectionUI):
    def __init__(self, window, parent=None):
        super(FixedSelection, self).__init__(parent)
        self.window = window
        self.ui = FixedSelectionUI()
        self.ui.setupUI(self, window)
        self.ui.timeEdit.setupMinMax(self.window.getMinAndMaxDateTime())
        self.toggleWindowOnTop(True) # Keeps window on top of main MagPy window

    def toggleWindowOnTop(self, val):
        self.setParent(self.window if val else None)
        dialogFlag = QtCore.Qt.Dialog
        if self.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = self.windowFlags()
        flags = flags | dialogFlag if val else flags & ~dialogFlag
        self.setWindowFlags(flags)

    def setTimeEdit(self, strt, end):
        self.ui.timeEdit.start.setDateTime(strt)
        self.ui.timeEdit.end.setDateTime(end)

    def getTimeEditValues(self):
        strt = self.ui.timeEdit.start.dateTime()
        end = self.ui.timeEdit.end.dateTime()
        return strt, end

    def closeEvent(self, ev):
        self.close()
        self.window.closeFixSelection()

class TimeRegionSelector(QtWidgets.QFrame):
    def __init__(self, window, parent=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.window = window
        self.setupLayout()
        self.updateBtn.clicked.connect(self.applySelection)

    def setupLayout(self):
        # Time edits and update button
        self.resize(300, 50)
        self.setWindowTitle('Time Select')
        layout = QtWidgets.QGridLayout(self)
        self.timeEdit = TimeEdit(QtGui.QFont())
        self.timeEdit.setupMinMax(self.window.getMinAndMaxDateTime())
        self.updateBtn = QtWidgets.QPushButton('Apply')
        layout.addWidget(self.timeEdit.start, 0, 0, 1, 1)
        layout.addWidget(self.timeEdit.end, 0, 1, 1, 1)
        layout.addWidget(self.updateBtn, 0, 2, 1, 1)

    def getTimes(self):
        return self.timeEdit.start.dateTime(), self.timeEdit.end.dateTime()

    def applySelection(self):
        t0, t1 = self.window.getTimeTicksFromTimeEdit(self.timeEdit)
        self.window.selectTimeRegion(t0, t1)
        self.window.closeTimeSelect()

class BatchSelectUI(BaseLayout):
    def setupUI(self, frame):
        self.frame = frame
        frame.setWindowTitle('Batch Select')
        frame.resize(425, 350)
        frame.move(1000, 100)

        layout = QtWidgets.QGridLayout(frame)
        timeLt = self.setupTimeSelect()
        layout.addLayout(timeLt, 0, 0, 1, 1)

    def setupTimeSelect(self):
        layout = QtWidgets.QGridLayout()

        # Set up user input text box
        placeText = '[2016 Feb 21 01:00:30.250, 2016 Feb 21 02:00:00.000], '
        placeText += '[2001 Mar 1 23:15:00.000, 2001 Mar 1 23:45:00.500]'

        self.inputBox = QtWidgets.QPlainTextEdit()
        self.inputBox.setPlaceholderText(placeText)
        self.inputBox.setWindowFlag(QtCore.Qt.Window)
        self.inputBox.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        inputLbl = QtWidgets.QLabel('Enter a list of UTC-formatted times:')
        inputLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Wrap in a frame
        textFrm = QtWidgets.QFrame()
        textLt = QtWidgets.QGridLayout(textFrm)
        textLt.setContentsMargins(0, 0, 0, 0)
        textLt.addWidget(inputLbl, 0, 0, 1, 1)
        textLt.addWidget(self.inputBox, 1, 0, 1, 1)

        # Set up item history box and add/rmv buttons
        self.regionList = TableWidget(2)
        self.regionList.setHeader(['Start Time', 'End Time'])
        self.regionList.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        btnLt = QtWidgets.QHBoxLayout()
        self.addBtn = QtWidgets.QPushButton('+')
        self.rmvBtn = QtWidgets.QPushButton('-')
        self.moveLowerBtn = QtWidgets.QPushButton('<')
        self.moveUpperBtn = QtWidgets.QPushButton('>')
        btnLt.addStretch()
        for btn in [self.addBtn, self.rmvBtn, self.moveLowerBtn, self.moveUpperBtn]:
            btn.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            btnLt.addWidget(btn)
        btnLt.addStretch()

        # Map remove connections
        self.rmvBtn.clicked.connect(self.removeItems)
        self.moveLowerBtn.clicked.connect(self.moveLower)
        self.moveUpperBtn.clicked.connect(self.moveUpper)

        # Keep On Top checkbox
        self.keepOnTopChk = QtWidgets.QCheckBox('Keep On Top')
        self.keepOnTopChk.setChecked(True)
        self.keepOnTopChk.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Status bar and highlight selections checkbox layout
        self.statusBar = QtWidgets.QStatusBar()
        self.highlightChk = QtWidgets.QCheckBox('Highlight Selections')
        self.highlightChk.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.highlightChk.setChecked(True)

        statusLt = QtWidgets.QHBoxLayout()
        statusLt.addWidget(self.highlightChk)
        statusLt.addWidget(self.statusBar)

        # Add everything to layout
        layout.addWidget(textFrm, 0, 0, 1, 1)
        layout.addLayout(btnLt, 1, 0, 1, 1)
        layout.addWidget(self.regionList, 2, 0, 1, 1)
        layout.addLayout(statusLt, 3, 0, 1, 1)

        # Set row stretch factors so input box is smaller than list view
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 5)

        return layout

    def removeItems(self):
        # Get selected rows, sort them, and convert to numpy array
        rows = self.regionList.getSelectedRows()
        rows.sort()
        origRows = rows[:]
        rows = np.array(rows)

        # Remove last row if nothing is selected
        if len(rows) == 0 and self.regionList.count() > 0:
            rows = [self.regionList.count()-1]

        # Remove each selected row
        for i in range(0, len(rows)):
            row = rows[i]
            self.removeItem(row)

            # Adjust indices for rows above current index that need to be removed
            # since they are no longer valid
            if i < (len(rows) - 1):
                rows[i:] -= 1

    def removeItem(self, row):
        if row >= 0:
            self.regionList.removeRow(row)
            self.frame.removeRegionFromGrid(row)
    
    def moveLower(self):
        row = self.getCurrentRow()
        if row <= 0:
            return
        self.regionList.setCurrentRow(row - 1)
    
    def moveUpper(self):
        row = self.getCurrentRow()
        if row >= self.regionList.count() - 1:
            return
        self.regionList.setCurrentRow(row + 1)

    def getCurrentRow(self):
        currRow = self.regionList.currentRow()
        count = self.regionList.count()
        # If there are items in list and none are selected, default is first row
        if count > 0 and currRow < 0:
            currRow = 0
        return currRow

class BatchSelect(QtWidgets.QFrame, BatchSelectUI):
    def __init__(self, window, parent=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.window = window

        self.ui = BatchSelectUI()
        self.ui.setupUI(self)
        self.ui.addBtn.clicked.connect(self.addRegionsToList)
        self.ui.regionList.currentCellChanged.connect(self.update)
        self.ui.highlightChk.toggled.connect(self.setRegionsVisible)

        self.linkedRegion = GeneralSelect(window, 'Multi', )
        self.linkedRegion.setLabelPos('bottom')
        self.linkedRegion.setMovable(False)
        self.setRegionsVisible(self.ui.highlightChk.isChecked())
        self.timeRegions = {} # Key = timestamp tuples, value = tick values
        self.regions = []

        # Linked update info
        self.linkedTimeEdit = None
        self.updateFunc = None

        # Timestamp regex string
        self.timeRegex = self.getTimestampRegex()

    def getTimestampRegex(self):
        # Get months in abbreviated format and generate regular expr for this group
        months = [datetime(2000, mon, 1).strftime('%b') for mon in range(1, 12+1)]
        monthRegex = '('+'|'.join(months)+')'

        # Year and day regular expressions
        yearRegex = '(19|20)[0-9]{2}'
        dayRegex = '([0-2][1-9]|[1-3][0-1])' # 1-9 ending, 0-1 ending

        # Hour, minute/seconds, milliseconds regular expressions
        hrRegex = '([0-1][0-9]|2[0-3])'
        minSecRegex = '[0-5][0-9]'
        msRegex = '[0-9]{3}'

        # Combine section regular expressions to form a UTC timestamp
        # regular expression
        dateExpr = ' '.join([yearRegex, monthRegex, dayRegex])
        timeExpr = hrRegex + ':' + minSecRegex + ':' + minSecRegex + '\.' + msRegex
        expr = dateExpr + ' ' + timeExpr

        return expr

    def setRegionsVisible(self, val):
        self.linkedRegion.setAllRegionsVisible(val)

    def numRegions(self):
        return self.ui.regionList.count()

    def addRegionsToList(self):
        txt = self.ui.inputBox.toPlainText()
        txt = txt.strip('\n')

        if len(txt) < 10:
            return

        # Regular expressing matching single pairs of matched brackets w/
        # commas between the brackets
        bracketExpr = '\[[^\[\]]+,[^\[\]]+\]'

        if not self.validateInputFormat(txt, bracketExpr):
            self.ui.statusBar.showMessage('Error: Input not formatted correctly')
            return

        # Split text by brackets
        pairs = re.findall(bracketExpr, txt)

        # Split text within brackets by commas and remove any extra lists
        pairs = [expr.strip('[').strip(']').split(',') for expr in pairs]
        pairs = [(pair[0].strip(' '), pair[1].strip(' ')) for pair in pairs if len(pair) == 2]

        # Separate out properly formatted timestamps
        correctPairs = []
        oddPairs = []
        for x, y in pairs:
            if self.validateTimestamp(x) and self.validateTimestamp(y):
                correctPairs.append((x,y))
            else:
                oddPairs.append((x,y))

        # Add regions to list and clear text box
        for x, y in correctPairs:
            self.addRegion(x, y)

        self.ui.inputBox.clear()
        self.ui.statusBar.clearMessage()

        # Add in improperly formatted timestamps back into input box
        if len(oddPairs) > 0:
            oddPairs = ['[' + x + ', ' + y + ']' for x, y in oddPairs]
            boxText = ', '.join(oddPairs)
            self.ui.inputBox.setPlainText(boxText)
            self.ui.statusBar.showMessage('Error: Timestamps not formatted correctly')
    
    def validateInputFormat(self, txt, bracketExpr):
        # Split everything outside of properly formatted brackets into groups
        groups = re.split(bracketExpr, txt)
        # Join groups together into a single string and check that it's only made up of
        # spaces and commas and newlines
        txt = ''.join(groups)
        chars = set(txt)
        allowedChars = set([',', ' ', '\n'])
        setDiff = chars - allowedChars
        if len(setDiff) > 0:
            return False
        return True
    
    def validateTimestamp(self, timestamp):
        # Check if timestamp fully matches regular expression
        if re.fullmatch(self.timeRegex, timestamp):
            return True
        return False

    def addRegion(self, startTime, endTime):
        pair = (startTime, endTime)
        if pair in self.timeRegions:
            return

        self.ui.regionList.addRowItem([startTime, endTime])
        self.addRegionToGrid(startTime, endTime)
        self.updateRegionLabels()
    
    def getRegion(self, rowNum):
        # Get item from interface and split text into times
        item = self.ui.regionList.removeRow(rowNum)
        text = item.text()
        splitText = item.split(' ')
        return (splitText[0], splitText[-1])

    def mapTimestamp(self, timestmp):
        # Map UTC timestamp into a time (library) tuple/struct
        splitStr = timestmp.split('.')
        if len(splitStr) > 1:
            dateStr, msStr = splitStr
        else:
            dateStr = splitStr[0]
            msStr = ''
        fmtStr = '%Y %b %d %H:%M:%S'
        ffFmtStr = '%Y %j %b %d %H:%M:%S'
        utcTime = time.strptime(dateStr, fmtStr)
        if msStr != '':
            msStr = '.' + msStr

        # Map the tuple back to a UTC timestamp in the format that
        # FFTime recognizes and then map this timestamp into a time tick
        ffUtcTime = time.strftime(ffFmtStr, utcTime)+msStr
        return ffUtcTime

    def addRegionToGrid(self, startTime, endTime):
        timestamps = (startTime, endTime)
        # Map timestamps to FFTime format and then to (ordered) time ticks
        startTime = self.mapTimestamp(startTime)
        endTime = self.mapTimestamp(endTime)
        tO = self.window.getTickFromTimestamp(startTime)
        tE = self.window.getTickFromTimestamp(endTime)
        tO, tE = min(tO, tE), max(tO, tE)

        # Create region and save time region info in dict/list
        self.linkedRegion.addFullRegion(tO, tE)
        self.timeRegions[timestamps] = (tO, tE)
        self.regions.append(timestamps)

    def removeRegionFromGrid(self, regionNum):
        # Get corresponding region and remove it from list + dictionary
        region = self.regions.pop(regionNum)
        del self.timeRegions[region]

        # Remove the linked region
        self.linkedRegion.removeRegion(regionNum)
        self.updateRegionLabels()

    def updateRegionLabels(self):
        for i in range(0, len(self.regions)):
            self.linkedRegion.setLabelText('Region ' + str(i+1), i)
    
    def getCurrentRegion(self):
        regNum = self.ui.getCurrentRow()
        if regNum < 0: # No regions in list
            return None
        return self.timeRegions[self.regions[regNum]]

    def update(self):
        # Get the currently selected region if there are any in the list
        region = self.getCurrentRegion()
        if not region:
            return
        if self.ui.regionList.currentRow() < 0:
            self.ui.regionList.blockSignals(True)
            self.ui.regionList.setCurrentRow(0)
            self.ui.regionList.blockSignals(False)
        
        self.updateByTimeEdit(self.window.ui.timeEdit, region)

        if self.linkedTimeEdit:
            self.updateByTimeEdit(self.linkedTimeEdit, region, self.updateFunc)
 
    def setUpdateInfo(self, timeEdit, updateFunc=None):
        self.linkedTimeEdit = timeEdit
        self.updateFunc = updateFunc

    def updateByTimeEdit(self, timeEdit, region, updateFunc=None):
        tO, tE = region

        startDt = self.window.getDateTimeFromTick(tO)
        endDt = self.window.getDateTimeFromTick(tE)
        timeEdit.start.setDateTime(startDt)
        timeEdit.end.setDateTime(endDt)

        if updateFunc:
            updateFunc()

    def closeEvent(self, ev):
        self.linkedRegion.closeAllRegions()
        self.window.closeBatchSelect()
        self.close()
