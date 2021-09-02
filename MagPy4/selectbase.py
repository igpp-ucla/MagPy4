from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .MagPy4UI import TimeEdit
import pyqtgraph as pg
from .pgextensions import LinkedRegion
import time as tm
from datetime import datetime
from .layouttools import BaseLayout, TableWidget, SplitterWidget
from fflib import ff_time
from .plotbase import MagPyViewBox

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

        if mode in ['Mulit', 'Adjusting']:
            self.maxSteps = -1

        self.regions = []
        self.lblPos = 'top'
        self.movable = True # Whether user can adjust the regions
        self.hidden = False

        # Additional functions to call once user has started selection and
        # after a region is fully selected
        self.stepFunc = None
        self.fullStepFunc = None
        self.post_func = None

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

    def leftClick(self, x, vb, ctrlPressed=False):
        # Show sub-regions in this plot if applicable
        index = self.get_plot_index(vb)
        if not ctrlPressed:
            self.setSubRegionsVisible(index)

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

    def get_plot_index(self, vb):
        ''' Returns plot index associated with viewbox '''
        plots = self.window.pltGrd.get_plots()
        index = 0
        i = 0
        for plot in plots:
            if plot.getViewBox() == vb:
                index = i
                break
            i += 1
        
        return index

    def rightClick(self, vb):
        # Hide sub regions in this plot
        index = self.get_plot_index(vb)
        self.setSubRegionsVisible(index, visible=False)

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

    def closeAllRegions(self, closeTool=True):
        # Close tool
        if closeTool and self.closeFunc:
            self.closeFunc()

        # Removes all regions from plot items
        for region in self.regions:
            region.removeRegionItems()
        self.regions = []

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

        if self.post_func:
            self.post_func()

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
        plts = self.window.pltGrd.get_plots()
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

    def getRegion(self, regionIndex):
        if regionIndex >= 0 and regionIndex < self.numRegions():
            region = self.regions[regionIndex]
            tick_range = region.getRegion()
            return tick_range
        return None
    
    def setActiveRegion(self, index):
        if index >= 0 and index < self.numRegions():
            region = self.regions[index]
        else:
            region = None

        if region:
            self.regionActivated(region)
            self.updateTimeEdit(region)

    def setPostFunc(self, post_func):
        self.post_func = post_func

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

class SelectableViewBox(MagPyViewBox):
    # Viewbox that calls its window's GeneralSelect object in response to clicks
    def __init__(self, window, plotIndex, *args, **kwds):
        super().__init__(*args, **kwds)
        self.window = window
        self.plotIndex = plotIndex
        self.add_menu_act = None
        self.setMouseEnabled(x=False, y=False)

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        pos = ev.scenePos()
        rect = self.sceneBoundingRect()

        if not rect.contains(pos):
            return

        # Apply left click to plot grid; Window manages behavior
        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)
        self.window.gridLeftClick(x, self.plotIndex, ctrlPressed)

    def onRightClick(self, ev):
        # Attempt to apply right click to plot grid
        res = self.window.gridRightClick(self.plotIndex)
        if not res: # No selections currently active, use default right click method
            pg.ViewBox.mouseClickEvent(self, ev)

    def mouseClickEvent(self, ev):
        if self.window is None:
            pg.ViewBox.mouseClickEvent(self, ev)
            return

        if ev.button() == QtCore.Qt.LeftButton:
            self.onLeftClick(ev)

        else: # asume right click i guess, not sure about middle mouse button click
            self.onRightClick(ev)

        ev.accept()

    # mouse drags for now just get turned into clicks, like on release basically, feels nicer
    # technically only need to do this for spectra mode but not being used otherwise so whatever
    def mouseDragEvent(self, ev, axis=None):
        if self.window is None:
            pg.ViewBox.mouseDragEvent(self, ev)
            return

        if ev.isFinish(): # on release
            if ev.button() == QtCore.Qt.LeftButton:
                self.onLeftClick(ev)
            elif ev.button() == QtCore.Qt.RightButton:
                self.onRightClick(ev)
        ev.accept()

    def wheelEvent(self, ev, axis=None):
        ev.ignore()
    
    def addMenuAction(self, act):
        self.add_menu_act = act
        self.menu.addAction(act)
    
    def rmvMenuAction(self):
        if self.add_menu_act:
            self.menu.removeAction(self.add_menu_act)
            self.add_menu_act = None

class FixedSelectionUI():
    def setupUI(self, Frame, window):
        Frame.resize(200, 50)
        Frame.setWindowTitle('Saved Region')
        Frame.move(1200, 0)

        # UI elements
        instr = 'Selected Time Range:'
        lbl = QtWidgets.QLabel(instr)
        self.timeEdit = TimeEdit()

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
        self.timeEdit = TimeEdit()
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

class TimeFormatWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupLayout()
        self.fmtInfo.clicked.connect(self.showFormatInfo)
        self.formatDesc = None

    def setupLayout(self):
        timeFmtLt = QtWidgets.QHBoxLayout(self)
        timeFmtLt.setContentsMargins(0, 0, 0, 0)

        # Label
        timeFmtLbl = QtWidgets.QLabel('Format: ')
        timeFmtLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Time format string
        self.timeFmt = QtWidgets.QLineEdit()
        self.timeFmt.setText('%Y %b %d %H:%M:%S')
        self.timeFmt.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))

        # Format description button
        self.fmtInfo = QtWidgets.QPushButton('?')
        self.fmtInfo.setMaximumWidth(25)

        for elem in [timeFmtLbl, self.timeFmt, self.fmtInfo]:
            timeFmtLt.addWidget(elem)

    def getLineEdit(self):
        ''' Returns line edit widget '''
        return self.timeFmt
    
    def text(self):
        ''' Returns text in line edit widget '''
        return self.timeFmt.text()

    def showFormatInfo(self):
        ''' Displays strftime formatting options in a separate window '''
        self.closeFormatInfo()

        # Create a read-only text view
        self.formatDesc = QtWidgets.QTextEdit()
        self.formatDesc.setReadOnly(True)
        self.formatDesc.setWindowTitle('Time Format Descriptions')
        self.formatDesc.resize(500, 400)

        # Get strftime documentation and set formatDesc's text
        doc = tm.strftime.__doc__
        start = doc.find('Commonly used')
        text = doc[start:]
        self.formatDesc.setPlainText(text)
        self.formatDesc.show()

    def closeFormatInfo(self):
        ''' Closes format description window '''
        if self.formatDesc:
            self.formatDesc.close()
            self.formatDesc = None

class TimeRangeData():
    def __init__(self, dates, ticks, fmt=None, epoch=None):
        self.dates = dates
        self.values = ticks
    
        # Default timestamp format
        if fmt is None:
            self.fmt = '%Y %b %m %H:%M:%S.%f'
        else:
            self.fmt = fmt

        # Epoch values if any
        self.epoch = epoch

    def datetimes(self):
        return self.dates

    def ticks(self):
        return self.values

    def timestamps(self, fmt=None):
        # Get format string
        fmt = self.fmt if fmt is None else fmt

        # Extract start/stop datetimes and map to timestamp
        # using given format
        start, stop = self.datetimes()
        start_ts = start.strftime(fmt)
        stop_ts = stop.strftime(fmt)

        return (start_ts, stop_ts)
    
    def __eq__(self, val):
        if isinstance(val, TimeRangeData):
            start, stop = self.datetimes()
            ref_start, ref_stop = val.datetimes()
            same_start = (start == ref_start)
            same_stop = (stop == ref_stop)
            return (same_start and same_stop)
        elif isinstance(val, tuple):
            return (val == self.datetimes())
        else:
            return False

class BatchSelectUI(BaseLayout):
    def setupUI(self, frame):
        self.frame = frame
        frame.setWindowTitle('Batch Select')
        frame.resize(425, 350)
        frame.move(1000, 100)

        self.fmtBox = None
        self.layout = self.setupTimeSelect()
        frameLt = QtWidgets.QGridLayout(frame)
        frameLt.addLayout(self.layout, 0, 0, 1, 1)

        # Connect input type box to function that updates layout
        self.inputTypeBox.currentTextChanged.connect(self.inputMethodChanged)

    def getTextInputFrame(self):
        # Set up format layout
        self.fmtBox = TimeFormatWidget()

        # Set up user input text box
        self.inputBox = QtWidgets.QPlainTextEdit()
        self.inputBox.setWindowFlag(QtCore.Qt.Window)
        self.inputBox.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Set the placeholder text to indicate format
        fmt = self.fmtBox.text()
        dt0 = datetime(2020, 1, 2, 10, 45).strftime(fmt)
        dt1 = datetime(2020, 1, 2, 13).strftime(fmt)
        placeText = f'{dt0}, {dt1}'
        dt0 = datetime(2020, 6, 12, 1, 15).strftime(fmt)
        dt1 = datetime(2020, 6, 12, 1, 20).strftime(fmt)
        placeText += f'\n{dt0}, {dt1}'
        self.inputBox.setPlaceholderText(placeText)

        # Add an 'instructions' label
        inputLbl = QtWidgets.QLabel("Enter a list of times and press the '+' button:")
        inputLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Wrap in a frame
        textFrm = QtWidgets.QFrame()
        textLt = QtWidgets.QGridLayout(textFrm)
        textLt.setContentsMargins(0, 0, 0, 0)
        textLt.addWidget(inputLbl, 0, 0, 1, 1)
        textLt.addWidget(self.inputBox, 1, 0, 1, 1)
        textLt.addWidget(self.fmtBox, 2, 0, 1, 1)

        return textFrm

    def getMouseInputFrame(self):
        # Set up a horizontal layout with a pair of time edits indicating the
        # start/end times for a region
        self.timeEdit = TimeEdit()
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
        mode = self.inputTypeBox.currentText()
        self.inputFrm = self.getInputFrame(mode)
        self.addBtn.setVisible(False)

        # Set up item history box
        self.table = TableWidget(2)
        self.table.setHeader(['Start Time', 'End Time'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(250)

        # Get status bar and keepOnTop checkbox layout
        statusLt = self.setupStatusBarLt()

        # Wrap table and options layouts in a separate widget/frame
        self.tableFrm = QtWidgets.QWidget()
        tableLt = QtWidgets.QVBoxLayout(self.tableFrm)
        tableLt.setContentsMargins(0, 0, 0, 0)
        tableLt.addLayout(btnLt)
        tableLt.addWidget(self.table)
        tableLt.addLayout(optionsLt)
        tableLt.addLayout(statusLt)

        # Add everything to layout
        layout.addLayout(inputTypeLt, 0, 0, 1, 1)
        layout.addWidget(self.inputFrm, 1, 0, 1, 1)
        layout.addWidget(self.tableFrm)

        return layout

    def setupSplitter(self):
        # Add items to splitter and adjust handle height
        splitLt = SplitterWidget()
        splitLt.setOrientation(QtCore.Qt.Vertical)
        splitLt.setChildrenCollapsible(False)
        splitLt.addWidget(self.inputFrm)
        splitLt.addWidget(self.tableFrm)
        splitLt.setHandleWidth(18)
        return splitLt

    def setupFrame(self, mode):
        # Remove previous widget from layout, if any
        widget = self.layout.itemAtPosition(1, 0)
        if widget is not None:
            widget = widget.widget()
            self.layout.removeWidget(widget)
        
        # Get new input widget
        self.inputFrm = self.getInputFrame(mode)

        # Adjust splitter layout or remove
        if mode == 'List':
            inputWidget = self.setupSplitter()
        else:
            inputWidget = self.inputFrm
            self.layout.addWidget(self.tableFrm, 2, 0, 1, 1)

        # Add new layout to main layout
        self.layout.addWidget(inputWidget, 1, 0, 1, 1)

        # Delete old widget
        if widget is not None:
            widget.deleteLater()
        
    def inputMethodChanged(self):
        # Get current input method
        mode = self.inputTypeBox.currentText()

        # Adjust layout and input widgets
        self.setupFrame(mode)

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
        ''' Remove selected rows from table or last item if none are selected '''
        # Get list of selected rows
        rows = self.table.getSelectedRows()

        # Remove last row if nothing is selected
        if len(rows) == 0 and self.table.count() > 0:
            rows = [self.table.count()-1]

        # Remove regions from plot grid
        for row in sorted(rows, reverse=True):
            self.frame.removeRegionFromGrid(row)

        # Remove rows from table
        self.table.removeRows(rows)

        # Update labels
        self.frame.updateRegionLabels()

    def moveLower(self):
        row = self.getCurrentRow()
        if row <= 0:
            return
        self.table.setCurrentRow(row - 1)
    
    def moveUpper(self):
        row = self.getCurrentRow()
        if row >= self.table.count() - 1:
            return
        self.table.setCurrentRow(row + 1)

    def getCurrentRow(self):
        currRow = self.table.currentRow()
        count = self.table.count()
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

        # Window flags settings
        self.normFlags = self.windowFlags()
        self.table_fmt = '%Y %b %d %H:%M:%S.%f'

        # Button and checkbox connections
        self.ui.addBtn.clicked.connect(self.addRegionsToList)
        self.ui.table.currentCellChanged.connect(self.update)
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

        self.inputModeChanged()

    def toggleWindowOnTop(self, val):
        # Keeps window on top of main window while user updates lines
        if val:
            flag = QtCore.Qt.WindowStaysOnTopHint
            flags = self.normFlags | flag
        else:
            flags = self.normFlags
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
        if self.lockChecked() or self.numRegions() < 1:
            return

        # Get the currently active linked region
        index = self.linkedRegion.getActiveRegion()
        if index is None or index >= self.numRegions():
            return

        # Get selection start/end ticks
        select_ticks = self.linkedRegion.getRegion(index)

        # Get table item's start/end tick values
        table_data = self.ui.table.getRowData(index)[0]
        table_ticks = table_data.ticks()

        # Skip if values are the same
        if table_ticks == select_ticks:
            return

        # Create table text and data
        timestamps, data = self.ticks_to_data(select_ticks)

        # Update the table item text and data
        self.ui.table.setRowItem(index, list(timestamps))
        self.ui.table.setRowData(index, [data, None])

    def ticks_to_timestamps(self, ticks):
        ''' Map ticks time range to strings '''
        start, stop = ticks
        start = ff_time.tick_to_ts(start, self.window.epoch)[:-3]
        stop = ff_time.tick_to_ts(stop, self.window.epoch)[:-3]

        # Remove day of year
        start = start[:4] + start[8:]
        stop = stop[:4] + stop[8:]

        return (start, stop)
    
    def setRegionsVisible(self, val):
        self.linkedRegion.setAllRegionsVisible(val)

    def numRegions(self):
        return self.ui.table.count()

    def addRegionsToList(self):
        # Read text from inputBox and split into lines
        txt = self.ui.inputBox.toPlainText().strip('\n')
        lines = txt.split('\n')

        # Return if no lines in text
        if len(lines) < 1:
            return
        
        # Separate out properly formatted timestamps
        correctPairs = []
        invalidPairs = []
        for line in lines:
            pair = self.mapInput(line)
            if pair is None:
                invalidPairs.append(line)
            else:
                correctPairs.append(pair)

        # Add regions to list and clear text box
        for x, y in correctPairs:
            # Map to strings before adding region
            x = x.strftime(self.table_fmt)[:-3]
            y = y.strftime(self.table_fmt)[:-3]
            self.addRegion(x, y)

        self.ui.inputBox.clear()
        self.ui.statusBar.clearMessage()

        # Add in improperly formatted timestamps back into input box
        if len(invalidPairs) > 0:
            boxText = '\n'.join(invalidPairs)
            self.ui.inputBox.setPlainText(boxText)
            self.ui.statusBar.showMessage('Error: Timestamps not formatted correctly')

    def mapInput(self, line):
        # Check if times are comma-separated
        delim = ','
        line = line.strip(delim)
        if delim not in line:
            return None
        
        # Split into start/stop times and remove whitespace
        start, stop = line.split(delim)[:2]
        start = start.strip(' ')
        stop = stop.strip(' ')

        # Get format and convert to datetimes
        fmt = self.ui.fmtBox.text()
        try:
            start_date = datetime.strptime(start, fmt)
            end_date = datetime.strptime(stop, fmt)
            return (start_date, end_date)
        except:
            return None

    def addRegion(self, startTime, endTime, addToGrid=True):
        # Sort times
        startTime, endTime = sorted([startTime, endTime])

        # Map time range to datetimes
        start_dt = datetime.strptime(startTime, self.table_fmt)
        end_dt = datetime.strptime(endTime, self.table_fmt)
        dates = (start_dt, end_dt)

        # Check if data was previously added and ignore if so
        for i in range(self.ui.table.count()):
            item = self.ui.table.getRowData(i)[0]
            if item.datetimes() == dates:
                return

        # Map regions to time ticks
        tO = ff_time.date_to_tick(start_dt, self.window.epoch)
        tE = ff_time.date_to_tick(end_dt, self.window.epoch)
        tO, tE = min(tO, tE), max(tO, tE)
        ticks = (tO, tE)

        # Update table with timestamps and data
        data = TimeRangeData(dates, ticks)
        self.ui.table.addRowItem([startTime, endTime], data=[data, None])

        # Add to plot grid if new region was not added through mouse selection
        if addToGrid:
            self.addRegionToGrid(tO, tE)

        # Update labels for all of the regions in the plot grid
        self.updateRegionLabels()

    def getRegion(self, rowNum):
        table_data = self.ui.table.getRowData(rowNum)[0]
        return table_data
    
    def getRegions(self, ticks=True):
        n = self.numRegions()
        table_data = [self.getRegion(i) for i in range(0, n)]
        if ticks:
            return [data.ticks() for data in table_data]
        else:
            return [data.datetimes() for data in table_data]

    def addRegionToGrid(self, tO, tE):
        # Create a new linked region for the added region
        ofst = self.window.tickOffset
        self.linkedRegion.addRegion(tO-ofst, tE-ofst)

    def removeRegionFromGrid(self, regionNum):
        # Remove the linked region
        self.linkedRegion.removeRegion(regionNum)

    def mouseAddedRegion(self):
        ticks = self.linkedRegion.getRegion(self.numRegions())
        start_ts, end_ts = self.ticks_to_timestamps(ticks)
        self.addRegion(start_ts, end_ts, False)
    
    def ticks_to_data(self, ticks):
        # Map ticks to timestamps
        timestamps = self.ticks_to_timestamps(ticks)

        # Map timestamps to datetimes
        datetimes = self.timestamps_to_datetimes(timestamps, self.table_fmt)

        # Create data item
        data = TimeRangeData(datetimes, ticks)

        return list(timestamps), data
    
    def timestamps_to_datetimes(self, timestamps, fmt):
        start_ts, end_ts = timestamps
        start_dt = datetime.strptime(start_ts, fmt)
        end_dt = datetime.strptime(end_ts, fmt)
        return (start_dt, end_dt)

    def updateRegionLabels(self):
        # Update region labels to match their row index in the table
        for i in range(0, self.numRegions()):
            self.linkedRegion.setLabelText(f' {i+1} ', i)
    
    def getCurrentRegion(self):
        regNum = self.ui.getCurrentRow()
        if regNum < 0: # No regions in list
            return None
        table_data = self.ui.table.getRowData(regNum)[0]
        return table_data.datetimes()
    
    def update(self, row=None):
        # Update active row
        if row is not None:
            self.linkedRegion.setActiveRegion(row)

        # Get the currently selected region if there are any in the list
        region = self.getCurrentRegion()
        if not region:
            return

        # Set current row to zero if none are currently selected
        if self.ui.table.currentRow() < 0:
            self.ui.table.blockSignals(True)
            self.ui.table.setCurrentRow(0)
            self.ui.table.blockSignals(False)

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
        # Extract region datetimes
        startDt, endDt = region

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
