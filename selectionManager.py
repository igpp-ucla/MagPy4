from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QSizePolicy
from MagPy4UI import TimeEdit
import pyqtgraph as pg
from dataDisplay import UTCQDate
from pyqtgraphExtensions import LinkedRegion
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

        # Link time edit to lines
        self.linkLinesToTimeEdit()

    def numRegions(self):
        return len(self.regions)

    def setTimeEdit(self, newTimeEdit):
        # Remove link to any previously set time edits and vice versa
        if self.timeEdit:
            self.timeEdit.removeLinkToSelect(self.updateLinesByTimeEdit)

        # Set new time edit and link lines to it if newTimeEdit is not None
        self.timeEdit = newTimeEdit
        self.linkLinesToTimeEdit()
        for region in self.regions:
            region.setLinkedTE(self.timeEdit)

    def linkLinesToTimeEdit(self):
        # Links changes in time edit datetime values to this selection's
        # function for updating lines to match the time edit values
        if self.timeEdit:
            self.timeEdit.linkToSelect(self.updateLinesByTimeEdit)

    def onPlotRemoved(self, plt):
        # For every linked region
        for region in self.regions:
            # Find this plot in its list of linked plots
            if plt in region.plotItems:
                index = region.plotItems.index(plt)
                # Then remove it from its list of plots and remove its corresp. 
                # region items
                region.plotItems.pop(index)
                region.regionItems.pop(index)

    def getSelectionInfo(self):
        regions = [region.getRegion() for region in self.regions]
        return (self.name, regions)

    def loadToolFromState(self, regions, tool, state):
        # Add in selected regions
        ofst = self.window.tickOffset
        for start, stop in regions:
            self.addRegion(start-ofst, stop-ofst, update=False)

        # Make any minor state updates
        self.updateTimeEdit()
        self.callUpdateFunc()

        # Load tool state
        if state is not None:
            tool.loadState(state)

        self.callOpenFunc()

    def isFullySelected(self):
        val = False
        if self.mode in ['Line', 'Adjusting'] and len(self.regions) >= 1:
            val = True
        elif len(self.regions) > 1:
            val = True
        elif len(self.regions) == 1:
            if not self.regions[-1].fixedLine:
                val = True

        return val

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

    def addRegion(self, x, y=None, update=True):
        # Adds a new region/line to all plots and connects it to the timeEdit
        linkRegion = self.addLinkedItem(x)
        self.regions.append(linkRegion)

        # If adding a single line, fix it so dragging only moves the line
        # instead of extending it
        if y is None:
            linkRegion.setFixedLine(True)
        else:
            linkRegion.setRegion((x, y))

        # Set default move / visibility properties
        linkRegion.setMovable(self.movable)
        linkRegion.setAllRegionsVisible(not self.hidden)

        # Call any open & update functions if necessary
        if update:
            if y is None:
                if self.mode in ['Line', 'Adjusting']:
                    self.openTool(linkRegion)
            else:
                self.openTool(linkRegion)

        # Link this region's 'activated' signal to regionActivated function
        linkRegion.sigRegionActivated.connect(self.regionActivated)

        # Link newly added region's lines to time edit
        self.regionActivated(linkRegion)

        return linkRegion

    def updateTimeEdit(self, region=None):
        if region is None:
            region = self.regions[-1]
        if self.timeEdit:
            region.updateTimeEditByLines(self.timeEdit)

    def openTool(self, region):
        self.updateTimeEdit()
        self.callOpenFunc()
        self.callUpdateFunc()

    def callOpenFunc(self):
        if self.func:
            self.openFunc()

    def callUpdateFunc(self):
        if self.updtFunc:
            if self.mode in ['Adjusting', 'Line']:
                self.updtFunc()
            elif len(self.regions) > 0 and not self.regions[-1].isFixedLine():
                self.updtFunc()

    def setUpdateFunc(self, f):
        self.updtFunc = f

    def setOpenFunc(self, f):
        self.func = f

    def addLinkedItem(self, x):
        # Initializes the linked region object
        plts = self.window.plotItems
        linkRegion = LinkedRegion(self.window, plts, (x, x), mode=self.name, 
            color=self.color, updateFunc=self.callUpdateFunc, linkedTE=self.timeEdit, 
            lblPos=self.lblPos)
        return linkRegion

    def updateLinesByTimeEdit(self):
        # Skip if no linked time edit or no regions selected
        if self.timeEdit is None or len(self.regions) < 1:
            return

        # Make sure connected region is still in set of regions
        region = self.timeEdit.getLinkedRegion()
        if region is None or region not in self.regions:
            self.timeEdit.setLinkedRegion(None) # Remove old linked region
            return

        # Get new time ticks + use old time ticks from region to determine
        # what order they should be in (w.r.t. line ordering)
        x0, x1 = region.getRegion()
        t0, t1 = self.window.getTimeTicksFromTimeEdit(self.timeEdit)
        if x1 >= x0:
            t0, t1 = t1, t0

        # Set line positions
        self.updateLinesPos(region, t0, t1)

    def updateLinesPos(self, region, t0, t1):
        # Set the linked region's start/stop values and call the update function
        region.setRegion((t0-self.window.tickOffset, t1-self.window.tickOffset))
        self.callUpdateFunc()

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
        region.setFixedLine(False)

        self.openTool(region)

    def setLabelText(self, txt, regionNum=None):
        # Set label text for all regions if none is specified
        if regionNum is None:
            for region in self.regions:
                region.setLabelText(txt)
        else:
            self.regions[regionNum].setLabelText(txt)

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

    def getRegionTicks(self):
        ticks = []
        for region in self.regions:
            ticks.append(region.getRegion())

        return ticks
    
    def regionActivated(self, region):
        if self.timeEdit:
            self.timeEdit.setLinkedRegion(region)

class BatchSelectRegions(GeneralSelect):
    '''
        Custom GeneralSelect object for BatchSelect tool:
        Keeps track of active region and makes calls to open and update functions
        differently than the default GeneralSelect implementation
    '''
    def __init__(self, window, mode, *args, **kwargs):
        GeneralSelect.__init__(self, window, mode, *args, **kwargs)
        self.activeRegion = None

    def addRegion(self, x, y=None, update=False):
        GeneralSelect.addRegion(self, x, y, update=False)
        self.callOpenFunc() # Adds region to Batch Select table

    def regionActivated(self, region):
        GeneralSelect.regionActivated(self, region)
        self.activeRegion = region

    def getActiveRegion(self):
        # Get active region and make sure it is still in list of regions
        if self.activeRegion is None or self.activeRegion not in self.regions:
            self.activeRegion = None
            return None
        else: # Otherwise, find the index corresponding to the active region
            regionIndex = self.regions.index(self.activeRegion)
            return regionIndex

    def extendRegion(self, x, region, ctrlPressed):
        # Extends previously added region (that hasn't been expanded yet)
        if self.regions == []:
            return

        x0, x1 = region.getRegion()
        region.setRegion((x0-self.window.tickOffset, x))
        region.setFixedLine(False)

        # Updates the item in the batch select table to match region times
        self.callUpdateFunc()
        self.updateTimeEdit()

    def updateLinesByTimeEdit(self):
        # Setting region by time edit should automatically remove
        # the fixed line property for batch select linked regions
        GeneralSelect.updateLinesByTimeEdit(self)
        if len(self.regions) > 0:
            self.regions[-1].setFixedLine(False)

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

        # Apply left click to plot grid; Window manages behavior
        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)
        self.window.gridLeftClick(x, self.plotIndex, ctrlPressed)

    def onRightClick(self, ev):
        # Attempt to apply right click to plot grid
        res = self.window.gridRightClick(self.plotIndex)
        if not res: # No selections currently active, use default right click method
            pg.ViewBox.mouseClickEvent(self, ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
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
        strt = self.ui.timeEdit.start.dateTime().toPyDateTime()
        end = self.ui.timeEdit.end.dateTime().toPyDateTime()
        return strt, end

    def getState(self):
        state = {}
        state['TimeEdit'] = self.getTimeEditValues()
        return state

    def loadState(self, state):
        start, end = state['TimeEdit']
        self.setTimeEdit(start, end)

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

        frameLt = QtWidgets.QGridLayout(frame)
        self.layout = self.setupTimeSelect()
        frameLt.addLayout(self.layout, 0, 0, 1, 1)

        # Connect input type box to function that updates layout
        self.inputTypeBox.currentTextChanged.connect(self.inputMethodChanged)

    def getTextInputFrame(self):
        # Set up user input text box
        self.inputBox = QtWidgets.QPlainTextEdit()
        self.inputBox.setWindowFlag(QtCore.Qt.Window)
        self.inputBox.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Set the placeholder text to indicate format
        placeText = '[2016 Feb 21 01:00:30.250, 2016 Feb 21 02:00:00.000], '
        placeText += '[2001 Mar 1 23:15:00.000, 2001 Mar 1 23:45:00.500]'
        self.inputBox.setPlaceholderText(placeText)

        # Add an 'instructions' label
        inputLbl = QtWidgets.QLabel('Enter a list of UTC-formatted times:')
        inputLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Wrap in a frame
        textFrm = QtWidgets.QFrame()
        textLt = QtWidgets.QGridLayout(textFrm)
        textLt.setContentsMargins(0, 0, 0, 0)
        textLt.addWidget(inputLbl, 0, 0, 1, 1)
        textLt.addWidget(self.inputBox, 1, 0, 1, 1)

        return textFrm

    def getMouseInputFrame(self):
        # Set up a horizontal layout with a pair of time edits indicating the
        # start/end times for a region
        self.timeEdit = TimeEdit(QtGui.QFont())
        self.timeEdit.setupMinMax(self.frame.window.getMinAndMaxDateTime())

        frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addStretch()
        layout.addWidget(self.timeEdit.start)
        layout.addWidget(self.timeEdit.end)
        layout.addStretch()

        return frame

    def getOptionsLt(self):
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Set up options checkboxes
        self.highlightChk = QtWidgets.QCheckBox(' Highlight')
        self.highlightChk.setChecked(True)

        self.lockCheck = QtWidgets.QCheckBox(' Lock')

        self.autoDisp = QtWidgets.QCheckBox(' Auto Display')
        self.autoDisp.setChecked(True)

        hltt = 'Shows selected regions in plot grid'
        locktt = 'Disables modifying selections by time edits or mouse clicks'
        disptt = 'Automatically sets the displayed region to the selected time region'

        boxes = [self.highlightChk, self.lockCheck, self.autoDisp]
        tooltips = [hltt, locktt, disptt]
        for chk, tt in zip(boxes, tooltips):
            chk.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            chk.setToolTip(tt)

        # Add items to layout
        lbl = 'Options: '
        self.addPair(layout, lbl, self.highlightChk, 0, 0, 1, 1)
        layout.addWidget(self.lockCheck, 0, 2, 1, 1)
        layout.addWidget(self.autoDisp, 0, 3, 1, 1)

        # Add in spacer
        spacer = QtWidgets.QSpacerItem(0, 0, hPolicy=QSizePolicy.Expanding)
        layout.addItem(spacer, 0, 4, 1, 1)

        return layout

    def getInputFrame(self, mode='List'):
        if mode == 'List':
            return self.getTextInputFrame()
        else:
            return self.getMouseInputFrame()

    def setupInputTypeLt(self):
        # Set up input type box
        self.inputTypeBox = QtWidgets.QComboBox()
        self.inputTypeBox.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.inputTypeBox.addItems(['List', 'Mouse Select'])
        self.inputTypeBox.setCurrentIndex(1)

        # Tooltip
        inputTt = 'Method for selecting times: \nList - Enter a list of UTC-'
        inputTt += 'formatted times\nMouse: Select time ranges directly on plot grid'

        # Add in combo box, its label, and a spacer item
        inputTypeLt = QtWidgets.QGridLayout()
        inputTypeLt.setContentsMargins(0, 0, 0, 0)

        lbl = 'Input Method: '
        self.addPair(inputTypeLt, lbl, self.inputTypeBox, 0, 0, 1, 1, inputTt)

        spacer = QtWidgets.QSpacerItem(0, 0, hPolicy=QSizePolicy.Expanding)
        inputTypeLt.addItem(spacer, 0, 2, 1, 1)

        return inputTypeLt

    def setupBtnLt(self):
        # Initialize buttons for navigating and modifying items in the
        # table widget
        self.addBtn = QtWidgets.QPushButton('+')
        self.rmvBtn = QtWidgets.QPushButton('–')
        self.moveLowerBtn = QtWidgets.QPushButton('<')
        self.moveUpperBtn = QtWidgets.QPushButton('>')

        # Add buttons to a horizontal layout and center them
        btnLt = QtWidgets.QHBoxLayout()
        btnLt.addStretch()
        for btn in [self.addBtn, self.rmvBtn, self.moveLowerBtn, self.moveUpperBtn]:
            btn.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            btnLt.addWidget(btn)
        btnLt.addStretch()

        # Map button connections
        self.rmvBtn.clicked.connect(self.removeItems)
        self.moveLowerBtn.clicked.connect(self.moveLower)
        self.moveUpperBtn.clicked.connect(self.moveUpper)

        return btnLt

    def setupStatusBarLt(self):
        # Keep On Top checkbox
        self.keepOnTopChk = QtWidgets.QCheckBox('Keep On Top')
        self.keepOnTopChk.setChecked(True)
        self.keepOnTopChk.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Status bar
        self.statusBar = QtWidgets.QStatusBar()

        # Add items to a horizontal layout
        statusLt = QtWidgets.QHBoxLayout()
        statusLt.addWidget(self.keepOnTopChk)
        statusLt.addWidget(self.statusBar)

        return statusLt

    def setupTimeSelect(self):
        layout = QtWidgets.QGridLayout()

        # Get input type box layout
        inputTypeLt = self.setupInputTypeLt()

        # Get selection options layout
        optionsLt = self.getOptionsLt()

        # Set up add/remove button layout
        btnLt = self.setupBtnLt()

        # Set up input frame (mouse-mode is default)
        self.inputFrm = self.getInputFrame(self.inputTypeBox.currentText())
        self.addBtn.setVisible(False)

        # Set up item history box
        self.regionList = TableWidget(2)
        self.regionList.setHeader(['Start Time', 'End Time'])
        self.regionList.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Get status bar and keepOnTop checkbox layout
        statusLt = self.setupStatusBarLt()

        # Add everything to layout
        layout.addLayout(inputTypeLt, 0, 0, 1, 1)
        layout.addWidget(self.inputFrm, 1, 0, 1, 1)
        layout.addLayout(optionsLt, 4, 0, 1, 1)
        layout.addLayout(btnLt, 2, 0, 1, 1)
        layout.addWidget(self.regionList, 3, 0, 1, 1)
        layout.addLayout(statusLt, 5, 0, 1, 1)

        return layout

    def inputMethodChanged(self):
        # Get current input method
        mode = self.inputTypeBox.currentText()

        # Get old frame and remove/delete it
        self.layout.removeWidget(self.inputFrm)
        self.inputFrm.deleteLater()

        # Generate new frame and place it in the layout
        self.inputFrm = self.getInputFrame(mode)
        self.layout.addWidget(self.inputFrm, 1, 0, 1, 1)

        # Enable/disable related buttons
        lockEnabled = True
        addEnabled = False
        if mode == 'List':
            lockEnabled = False
            addEnabled = True

        self.lockCheck.setEnabled(lockEnabled)
        self.addBtn.setVisible(addEnabled)

        # Modify state in main frame
        self.frame.inputModeChanged()

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
            self.frame.rmvRegion(row)
    
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

        # Button and checkbox connections
        self.ui.addBtn.clicked.connect(self.addRegionsToList)
        self.ui.regionList.currentCellChanged.connect(self.update)
        self.ui.highlightChk.toggled.connect(self.setRegionsVisible)
        self.ui.lockCheck.toggled.connect(self.lockSelections)

        self.ui.keepOnTopChk.clicked.connect(self.toggleWindowOnTop)
        self.toggleWindowOnTop(self.ui.keepOnTopChk.isChecked())

        # Initialize linked region and state information lists
        self.linkedRegion = BatchSelectRegions(window, 'Multi', 
            func=self.mouseAddedRegion, maxSteps=-1)
        self.linkedRegion.setLabelPos('bottom')
        self.linkedRegion.setMovable(False)
        self.setRegionsVisible(self.highlightChecked())
        self.timeRegions = {} # Key = timestamp tuples, value = tick values
        self.regions = []

        # Linked update info
        self.linkedTimeEdit = None
        self.updateFunc = None

        # Timestamp regex string
        self.timeRegex = self.getTimestampRegex()
        self.inputModeChanged()

    def toggleWindowOnTop(self, val):
        # Keeps window on top of main window while user updates lines
        dialogFlag = QtCore.Qt.Dialog
        if self.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = self.windowFlags()
        flags = flags | dialogFlag if val else flags & ~dialogFlag
        self.setWindowFlags(flags)
        self.show()

    def lockSelections(self, val):
        # Disable mouse drags + line moving for linked regions if locked
        self.linkedRegion.setMovable(not val)

        if self.getInputMode() != 'List':
            if val: # Remove link to time edit if locked
                self.linkedRegion.setTimeEdit(None)
            else: # Set link to time edit if unlocked
                self.linkedRegion.setTimeEdit(self.ui.timeEdit)

    def getInputMode(self):
        return self.ui.inputTypeBox.currentText()

    def lockChecked(self):
        return self.ui.lockCheck.isChecked()

    def highlightChecked(self):
        return self.ui.highlightChk.isChecked()

    def autoDispChecked(self):
        return self.ui.autoDisp.isChecked()

    def isLocked(self):
        # Used by main window to determine whether mouse drags/clicks
        # should have an effect on the batch select linked region
        if self.getInputMode() == 'List':
            return True
        else: # Ignore if not visible or lock is checked
            locked = self.lockChecked()
            visible = self.highlightChecked()
            res = (locked or (not visible))
            return res

    def inputModeChanged(self):
        # Default settings for list mode, regions are locked and unlinked
        newTimeEdit = None
        updtFunc = None
        openFunc = None
        lockSelect = True

        # Link region to time edit and updateTableRegions function
        if self.getInputMode() != 'List':
            newTimeEdit = self.ui.timeEdit
            lockSelect = False
            updtFunc = self.updateTableRegions
            openFunc = self.mouseAddedRegion

        # Take into account whether lock checkbox is checked
        lockSelect = (self.lockChecked() and lockSelect)
        self.lockSelections(lockSelect)

        # Update the linked region's settings
        self.linkedRegion.setOpenFunc(openFunc)
        self.linkedRegion.setUpdateFunc(updtFunc)
        self.linkedRegion.setTimeEdit(newTimeEdit)

    def updateTableRegions(self):
        # Do not modify if regions are locked or there are no regions selected
        if self.lockChecked() or self.linkedRegion.numRegions() < 1:
            return

        # Get the currently active linked region
        regionNum = self.linkedRegion.getActiveRegion()
        if regionNum is None:
            return

        # Map region to timestamps/ticks
        regionItem = self.linkedRegion.regions[regionNum]
        ticks, timestamps = self.mapRegionToInfo(regionItem)
        oldTimestamps = self.regions[regionNum]

        # Do not modify table if region is the same as before
        if oldTimestamps == timestamps:
            return

        # Remove region from internal state and add new region info in its place
        self.rmvRegion(regionNum, rmvFromGrid=False)
        self.timeRegions[timestamps] = ticks
        self.regions.insert(regionNum, timestamps)

        # Update the table item
        self.ui.regionList.setRowItem(regionNum, list(timestamps))
    
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

    def addRegion(self, startTime, endTime, addToGrid=True):
        # Check if region was previously added; If so, ignore it
        pair = (startTime, endTime)
        if pair in self.timeRegions:
            return

        # Update table
        self.ui.regionList.addRowItem([startTime, endTime])

        # Internal state updates
        # Map regions to formatted timestamps and time ticks
        timestamps = (startTime, endTime)
        startTime = self.mapTimestamp(startTime)
        endTime = self.mapTimestamp(endTime)
        tO = self.window.getTickFromTimestamp(startTime)
        tE = self.window.getTickFromTimestamp(endTime)
        tO, tE = min(tO, tE), max(tO, tE)

        self.timeRegions[timestamps] = (tO, tE)
        self.regions.append(timestamps)

        # Add to plot grid if new region was not added through mouse selection
        if addToGrid:
            self.addRegionToGrid(tO, tE)

        # Update labels for all of the regions in the plot grid
        self.updateRegionLabels()

    def rmvRegion(self, regionNum, rmvFromGrid=True):
        # Get corresponding region and remove it from list + dictionary
        region = self.regions.pop(regionNum)
        del self.timeRegions[region]

        if rmvFromGrid:
            self.removeRegionFromGrid(regionNum)

    def getRegion(self, rowNum):
        return self.timeRegions[self.regions[rowNum]]
    
    def getRegions(self, ticks=True):
        if ticks:
            return [self.getRegion(rowNum) for rowNum in range(0, len(self.regions))]
        else:
            return self.timeRegions

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

    def addRegionToGrid(self, tO, tE):
        # Create a new linked region for the added region
        ofst = self.window.tickOffset
        self.linkedRegion.addRegion(tO-ofst, tE-ofst)

    def removeRegionFromGrid(self, regionNum):
        # Remove the linked region
        self.linkedRegion.removeRegion(regionNum)
        self.updateRegionLabels()

    def mapRegionToInfo(self, region):
        # Map linked region selection to timestamps and time ticks
        start, stop = region.getRegion()
        fmtTime = lambda s : s[0:5] + s[9:]
        startTime = self.window.getTimestampFromTick(start)
        stopTime = self.window.getTimestampFromTick(stop)
        startTime, stopTime = fmtTime(startTime), fmtTime(stopTime)
        timestamps = (startTime, stopTime)
        return (start, stop), timestamps

    def mouseAddedRegion(self):
        region = self.linkedRegion.regions[-1]
        (start, stop), (startTime, stopTime) = self.mapRegionToInfo(region)
        self.addRegion(startTime, stopTime, False)

    def updateRegionLabels(self):
        # Update region labels to match their row index in the table
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

        # Set current row to zero if none are currently selected
        if self.ui.regionList.currentRow() < 0:
            self.ui.regionList.blockSignals(True)
            self.ui.regionList.setCurrentRow(0)
            self.ui.regionList.blockSignals(False)

        # Set view range in the plot grid to current selection
        if self.autoDispChecked():
            self.updateByTimeEdit(self.window.ui.timeEdit, region)

        # Update the time edit for the currently linked tool, if there is one,
        # and also call its update function
        if self.linkedTimeEdit:
            self.updateByTimeEdit(self.linkedTimeEdit, region, self.updateFunc)
 
    def setUpdateInfo(self, timeEdit, updateFunc=None):
        self.linkedTimeEdit = timeEdit
        self.updateFunc = updateFunc
    
    def updateByTimeEdit(self, timeEdit, region, updateFunc=None):
        # Extract region and map to datetime objects
        tO, tE = region
        startDt = self.window.getDateTimeFromTick(tO)
        endDt = self.window.getDateTimeFromTick(tE)

        # Set time edit values
        timeEdit.start.setDateTime(startDt)
        timeEdit.end.setDateTime(endDt)

        # Call an update function if necessary
        if updateFunc:
            updateFunc()

    def getState(self):
        state = {}
        state['Regions'] = self.getRegions(ticks=False)
        state['Options'] = self.getOptionsState()
        state['Mode'] = self.getInputMode()
        return state

    def getOptionsState(self):
        state = {}
        state['KeepOnTop'] = self.ui.keepOnTopChk.isChecked()
        state['Lock'] = self.lockChecked()
        state['Highlight'] = self.highlightChecked()
        state['AutoDisplay'] = self.autoDispChecked()
        return state

    def loadState(self, state):
        # Temporarily set input mode to list so items are not added twice
        # because of signals
        self.ui.inputTypeBox.setCurrentText('List')

        # Add all previous regions
        regions = state['Regions']
        for start, stop in regions:
            self.addRegion(start, stop)

        # Set correct input mode here
        if 'Mode' in state:
            self.ui.inputTypeBox.setCurrentText(state['Mode'])

        # Set selection options
        if 'Options' in state:
            boxes = [self.ui.keepOnTopChk, self.ui.lockCheck, self.ui.highlightChk,
                self.ui.autoDisp]
            kws = ['KeepOnTop', 'Lock', 'Highlight', 'AutoDisplay']
            for kw, box in zip(kws, boxes):
                box.setChecked(state['Options'][kw])

    def closeEvent(self, ev):
        self.linkedRegion.closeAllRegions()
        self.window.closeBatchSelect()
        self.close()
