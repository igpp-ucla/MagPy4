
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg

import functools
from .MagPy4UI import PyQtUtils
from . import getRelPath
import os
import numpy as np

class TraceInfo():
    ''' Object containing information about a plot variable '''
    def __init__(self, name, edit, window):
        self.name = name
        self.edit = edit
        self.window = window

    def getTuple(self):
        return (self.name, self.edit)

    def getLabel(self):
        if self.edit < 0:
            return self.name
        else:
            return self.window.getLabel(self.name, self.edit)

    def __eq__(self, other):
        return (self.getTuple() == other.getTuple())

class VariableListItem(QtWidgets.QListWidgetItem):
    ''' Custom list item w/ TraceInfo set as data '''
    def __init__(self, info):
        label = info.getLabel()
        QtWidgets.QListWidgetItem.__init__(self, label)
        self.setData(QtCore.Qt.UserRole, info)

        # If plotting a spectrogram or other special plot,
        # set the text color to blue
        if info.edit < 0:
            color = pg.mkColor(0, 0, 255)
            self.setForeground(color)

    def getInfo(self):
        return self.data(QtCore.Qt.UserRole)

class DragPixmap(QtGui.QPixmap):
    ''' Drag handle pixmap icon '''
    def __init__(self):
        # Initialize pixmap with fixed size rect
        rect = QtCore.QRect(0, 0, 28, 20)
        QtGui.QPixmap.__init__(self, rect.width(), rect.height())

        # Make background transparent
        self.fill(QtCore.Qt.transparent)

        # Create a painter w/ grey pen
        p = QtGui.QPainter(self)
        pen = pg.mkPen((150, 150, 150))
        pen.setWidthF(1.3)
        p.setPen(pen)

        # Draw each line in rect
        padding = 2
        height = rect.height() - padding*2
        for i in range(0, 3):
            y = padding*1.5 + height * (i/3)
            p.drawLine(0, y, rect.width(), y)

        # End painter
        p.end()

class DragLabel(QtGui.QLabel):
    ''' Drag handle widget facilitates drag events for the given widget '''
    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        QtGui.QLabel.__init__(self, *args, **kwargs)
        self.pixmap = DragPixmap()
        self.setPixmap(self.pixmap)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setMaximumWidth(30)

    def mousePressEvent(self, ev):
        ''' Call widget's moveStarted function '''
        self.widget.moveStarted()

    def mouseMoveEvent(self, ev):
        ''' Move widget '''
        # Map position to parent coordinates
        pos = ev.pos()
        pos = QtCore.QPoint(pos.x(), pos.y() - self.rect().height())
        pos = self.mapToParent(pos)
        self.widget.moveToPos(pos)

    def mouseReleaseEvent(self, ev):
        ''' Call widget's moveFinished function '''
        self.widget.moveFinished()

class DragLayout(QtWidgets.QGridLayout):
    ''' Manages plot widgets in layout '''
    def __init__(self, *args, **kwargs):
        QtWidgets.QGridLayout.__init__(self, *args, **kwargs)

    def getItems(self):
        ''' Returns item in order of their row indices '''
        # Get all items
        count = self.count()
        items = [self.itemAtPosition(index, 0) for index in range(0, count)]

        # Update indices for each item
        index = 0
        for item in items:
            if item.widget().index == None:
                item.widget().index = index
            index += 1

        return items

    def getSnapPositions(self):
        ''' Get current positions of all widgets in layout '''
        positions = []
        for item in self.getItems():
            pos = item.widget().pos()
            positions.append(pos)
        return positions

    def updateZOrder(self, topWidget):
        ''' Stacks all widgets in layout underneath the given topWidget '''
        items = self.getItems()
        for item in items:
            if item.widget() != topWidget:
                item.widget().stackUnder(topWidget)

    def updateSnapPositions(self, topWidget):
        # Get items and sort by index value
        items = self.getItems()
        sortOrder = [item.widget().index for item in items]
        items = [items[i] for i in sortOrder]

        # Remove items
        for item in items:
            self.removeItem(item)

        # Place in correct order
        for i in range(0, len(items)):
            item = items[i]
            self.addItem(item, i, 0)

        # Update plot widget titles
        self.updateIndices()

    def findItem(self, widget):
        ''' Finds the row index for the given widget in the layout '''
        index = 0
        for item in self.getItems():
            if item.widget() == widget:
                break

            index += 1

        return index

    def updateIndices(self):
        ''' Update the plot label for each widget in layout '''
        items = self.getItems()
        for i in range(0, len(items)):
            items[i].widget().label.setText(f'Plot {i+1}:')

class PlotListWidget(QtWidgets.QFrame):
    def __init__(self, parentLt):
        self.parentLt = parentLt
        self.snapPositions = []
        self.index = None

        QtWidgets.QFrame.__init__(self)

        # Set up list view
        self.table = QtWidgets.QListWidget()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Set up label, drag hangle, and menu
        self.label = QtWidgets.QLabel('')
        self.dragHandle = DragLabel(self)
        self.menuBtn = self.setupMenu()

        titleLt = QtWidgets.QHBoxLayout()
        for elem in [self.label, self.dragHandle, self.menuBtn]:
            titleLt.addWidget(elem)

        # Set up buttons from adding/removing dstrs from plots
        self.addBtn = QtWidgets.QPushButton('>>')
        self.rmvBtn = QtWidgets.QPushButton(' < ')
        self.rmvBtn.clicked.connect(self.removeSelectedItems)

        ## Set up button layout
        btnLt = QtWidgets.QVBoxLayout()
        btnLt.addStretch()
        for btn in [self.addBtn, self.rmvBtn]:
            btn.setFixedWidth(50)
            btnLt.addWidget(btn)
        btnLt.addStretch()

        # Add items to layout
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        self.layout.addLayout(titleLt, 0, 1, 1, 1)
        self.layout.addLayout(btnLt, 1, 0, 1, 1)
        self.layout.addWidget(self.table, 1, 1, 1, 1)

        # Set a non-transparent background
        self.setStyleSheet('''
            PlotListWidget { background-color: rgba(240, 240, 240, 255); 

                }
            ''')

    def setupMenu(self):
        # Set up sub-plot options button
        optBtn = QtWidgets.QPushButton()
        optBtn.setStyleSheet('QPushButton { color: #292929; }')
        optBtn.setFlat(True)
        optBtn.setFixedSize(20, 20)

        # Set up options menu and 'close' action
        menu = QtWidgets.QMenu()
        self.removeAction = menu.addAction('Remove sub-plot')
        optBtn.setMenu(menu)

        # Set up height factor action + its internal layout
        frame = QtWidgets.QFrame()
        lt = QtWidgets.QGridLayout(frame)
        lt.setContentsMargins(30, 2, 2, 2)

        lbl = QtWidgets.QLabel('Set height factor: ')
        lt.addWidget(lbl, 0, 0, 1, 1)

        self.heightBox = QtWidgets.QSpinBox()
        self.heightBox.setMinimum(1)
        self.heightBox.setMaximum(5)
        lt.addWidget(self.heightBox, 0, 1, 1, 1)

        # Add action to menu
        act = QtWidgets.QWidgetAction(menu)
        act.setDefaultWidget(frame)
        menu.addSeparator()
        menu.addAction(act)

        return optBtn

    def getItems(self):
        ''' Returns a list of items in table format '''
        items = [self.table.item(row) for row in range(self.table.count())]
        return items

    def addItems(self, items):
        ''' Adds non-duplicate items to table widget '''
        # Get items (in tuple format) currently in table
        currItems = self.getItems()
        currItems = [item.getInfo().getTuple() for item in currItems]

        # Add in items if not currently in table
        for item in items:
            pair = item.getInfo().getTuple()
            if pair not in currItems:
                newItem = VariableListItem(item.getInfo())
                self.table.addItem(newItem)

    def removeItems(self, items):
        ''' Removes given items from table '''
        # Get current items and items to remove
        rmvItems = [item.getInfo().getTuple() for item in items]
        currItems = self.getItems()

        # Get items that aren't in remove list
        newItems = []
        for item in currItems:
            if item.getInfo().getTuple() not in rmvItems:
                newItem = VariableListItem(item.getInfo())
                newItems.append(newItem)

        # Clear table and re-add items
        self.table.clear()
        self.addItems(newItems)

    def removeSelectedItems(self):
        ''' Removes currently selected items from table '''
        items = self.table.selectedItems()
        self.removeItems(items)

    def moveStarted(self):
        # Save positions for items to snap back to
        self.snapPositions = self.parentLt.getSnapPositions()

        # Update z order so this widget has a higher z value
        self.parentLt.updateZOrder(self)

        # Save the index this item is currently at and make sure
        # other indices are set
        self.index = self.parentLt.findItem(self)
        self.parentLt.getItems()

    def moveToPos(self, pos):
        # Map position to parent coordinates
        pos = self.mapToParent(pos)
        pos = QtCore.QPoint(0, pos.y())

        # Move widget to new position
        self.move(pos)

        # Get position ranges for each widget
        bins = self.getBins()

        # Look for the new index corresponding to this widget
        y = pos.y()
        for i in range(0, len(bins)-1):
            b = bins[i]
            b_next = bins[i+1]

            # If current position falls in bin, switch indices with
            # widget in current index
            if i != self.index and (y > b) and (y < b_next):
                # Update indices and have parent layout update positions
                items = self.parentLt.getItems()
                items[i].widget().index = self.index
                self.index = i
                self.parentLt.updateSnapPositions(self)
                break

    def getBins(self):
        ''' Gets bins for falling into another row in grid '''
        # Get original positions
        yPos = [pos.y() for pos in self.snapPositions]

        # Get halfway points between each y-position
        diff = yPos[1] - yPos[0]
        halfPoints = np.array(yPos) + (diff/2)

        # Add in a zero at beginning to indicate first bin
        halfPoints = np.concatenate([[0], halfPoints])
        return halfPoints

    def moveFinished(self):
        # Remove and re-add widget to layout to snap it back
        # into the correct position
        index = self.parentLt.findItem(self)
        self.parentLt.removeWidget(self)
        self.parentLt.addWidget(self, index, 0)

class AdjustingScrollArea(QtWidgets.QScrollArea):
    # Scroll area that hides border when vertical scrollbar is not visible
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        self.setAcceptDrops(True)

    def resizeEvent(self, ev):
        QtWidgets.QScrollArea.resizeEvent(self, ev)
        self.updateFrameBorder()

    def updateFrameBorder(self):
        if self.verticalScrollBar().isVisible():
            self.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.setStyleSheet('QScrollArea { background-color : #ffffff; }')
        else:
            self.setFrameShape(QtWidgets.QFrame.NoFrame)

class ListLayout(QtWidgets.QGridLayout):
    def __init__(self, window, frame):
        self.window = window
        self.mainFrame = frame
        self.pltFrms = []
        self.pltTbls = []
        self.editNumbers = []

        QtWidgets.QGridLayout.__init__(self)
        self.setupLayout()

    def setupLayout(self):
        # Sets up dstr selector and plot selections layout
        layout = self

        self.dstrTableFrame, self.dstrTable = self.setupDstrTable(layout)
        layout.addWidget(self.dstrTableFrame, 0, 0, 1, 1)

        self.plotWrappers = []
        self.pltFrms, self.pltTbls, self.elems = [], [], []
        # self.pltLtFrm = DragFrameArea()
        self.pltLtFrm = QtWidgets.QFrame()
        self.pltLt = DragLayout(self.pltLtFrm)
        self.pltLt.setContentsMargins(0, 0, 0, 0)

        # Set up scroll area wrapper for plot boxes
        self.scrollArea = AdjustingScrollArea()
        testWidget = PlotListWidget(self.pltLt)
        width = testWidget.sizeHint().width()
        height = testWidget.table.sizeHint().height()

        # self.scrollArea.setMinimumWidth(width+10)
        # self.scrollArea.setMinimumHeight(400)
        self.scrollArea.setWidget(self.pltLtFrm)
        layout.addWidget(self.scrollArea, 0, 1, 1, 1)

    def setupDstrTable(self, layout):
        frame = QtWidgets.QGroupBox('Data Variables')
        layout = QtWidgets.QVBoxLayout(frame)
        table = QtWidgets.QListWidget()

        dstrs = self.window.DATASTRINGS
        infos = [TraceInfo(dstr, 0, self.window) for dstr in dstrs]
        for info in infos:
            table.addItem(VariableListItem(info))

        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        layout.addWidget(table)
        return frame, table

    def getPlotTitleLt(self, plotNum):
        # Set up plot number title
        pltTitleLt = QtWidgets.QHBoxLayout()
        pltLbl = QtWidgets.QLabel('Plot ' + str(plotNum) + ':')

        # Set up sub-plot options button
        optBtn = QtWidgets.QPushButton()
        optBtn.setStyleSheet('QPushButton { color: #292929; }')
        optBtn.setFlat(True)
        optBtn.setFixedSize(20, 20)

        # Set up options menu and 'close' action
        menu = QtWidgets.QMenu()
        act = menu.addAction('Remove sub-plot')
        act.triggered.connect(functools.partial(self.subMenuAction, optBtn, 'Remove'))
        act = menu.addAction('Move up')
        act.triggered.connect(functools.partial(self.subMenuAction, optBtn, 'Up'))
        act = menu.addAction('Move down')
        act.triggered.connect(functools.partial(self.subMenuAction, optBtn, 'Down'))
        optBtn.setMenu(menu)

        # Set up height factor action + its internal layout
        frame = QtWidgets.QFrame()
        lt = QtWidgets.QGridLayout(frame)
        lt.setContentsMargins(30, 2, 2, 2)

        lbl = QtWidgets.QLabel('Set height factor: ')
        lt.addWidget(lbl, 0, 0, 1, 1)

        heightBox = QtWidgets.QSpinBox()
        heightBox.setMinimum(1)
        heightBox.setMaximum(5)
        lt.addWidget(heightBox, 0, 1, 1, 1)

        act = QtWidgets.QWidgetAction(menu)
        act.setDefaultWidget(frame)
        menu.addSeparator()
        menu.addAction(act)

        # Set up drag and drop label item
        dragLbl = DragLabel()

        # Create layout
        pltElems = [pltLbl, dragLbl, optBtn]
        for elem in pltElems:
            pltTitleLt.addWidget(elem)

        return pltTitleLt, [pltLbl, optBtn] + [heightBox]

    def addPlot(self):
        plotNum = self.pltLt.count()
        plotFrame = PlotListWidget(self.pltLt)
        plotFrame.index = plotNum
        plotFrame.label.setText(f'Plot {plotNum+1}:')
        self.pltLt.addWidget(plotFrame, plotNum, 0, 1, 1)
        plotFrame.addBtn.clicked.connect(functools.partial(self.addDstrsToPlt, plotFrame))
        plotFrame.removeAction.triggered.connect(functools.partial(self.removeSpecificPlot, plotFrame))

        return plotFrame

    def plotCount(self):
        return len(self.pltLt.getItems())

    def subPlotAction(self, btn):
        # Try to find the index and function corresp. to given button
        index = 0
        for addBtn, rmvBtn, lbl, optBtn, heightBox in self.elems:
            if addBtn == btn:
                self.addDstrsToPlt(index)
                break
            elif rmvBtn == btn:
                self.rmvDstrsFrmPlt(index)
                break
            index += 1

    def setHeightFactor(self, plotIndex, factor):
        items = [item.widget() for item in self.pltLt.getItems()]
        items[plotIndex].heightBox.setValue(factor)

    def removePlot(self):
        # Remove plot layout from layout and delete the plot elements
        items = [item for item in self.pltLt.getItems()]
        item = items[-1]
        self.pltLt.removeItem(item)
        item.widget().deleteLater()


    def removeSpecificPlot(self, item):
        # Remove plot layout from layout and delete the plot elements
        index = None
        items = self.pltLt.getItems()
        i = 0
        for elem in items:
            if elem.widget() == item:
                index = i
                break
            i += 1

        if index is None:
            return

        self.pltLt.removeItem(items[index])
        item.deleteLater()

        # Shift all plot layouts above this index down by one index
        for row in range(index+1, len(items)):
            item = items[row]
            self.pltLt.removeItem(items[row])
            self.pltLt.addWidget(item.widget(), row-1, 0, 1, 1)
            item.widget().label.setText(f'Plot {row}:')

    def mapItems(self, lst):
        items = []
        for dstr, en in lst:
            info = TraceInfo(dstr, en, self.window)
            item = VariableListItem(info)
            items.append(item)
        return items

    def initPlots(self, dstrLst):
        # Creates a table for each dstr sub-list and fills it w/ the given dstrs
        for subLst in dstrLst:
            pltFrm = self.addPlot()
            items = self.mapItems(subLst)
            pltFrm.addItems(items)

    def clearPlots(self):
        items = self.pltLt.getItems()
        for item in items:
            item.widget().table.clear()

    def updtDstrOptions(self):
        # Update dstr table w/ current edit's dstrs
        self.dstrTable.clear()
        editNum = self.mainFrame.getCurrentEdit()
        for k,v in self.window.DATADICT.items():
            if len(v[editNum]) > 0:
                info = TraceInfo(k, editNum, self.window)
                item = VariableListItem(info)
                self.dstrTable.addItem(item)

    def addDstrsToPlt(self, plotWidget):
        # Adds the selected dstrs from the dstr table to the given plot table
        selectedItems = self.dstrTable.selectedItems()
        plotWidget.addItems(selectedItems)

    def getPltDstrs(self, pltNum):
        ''' Extra plot trace information from each plot frame '''
        plts = self.pltLt.getItems()
        groups = []
        for plt in plts:
            items = plt.widget().getItems()
            items = [item.getInfo().getTuple() for item in items]
            groups.append(items)

        return groups

    def getHeightFactors(self):
        items = [item.widget() for item in self.pltLt.getItems()]
        factors = []
        for item in items:
            factors.append(item.heightBox.value())
        return factors

class PlotMenuUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Plot Menu')

        base_url = getRelPath('images', directory=True)

        # Build icon image style sheet for each case
        checkBoxStyles = ['QCheckBox{spacing: 0px;}', 
            'QCheckBox::indicator{width:28px;height:28px}']

        for kw in ['unchecked', 'checked', 'indeterminate']:
            for subKw in ['', 'hover', 'pressed']:
                # Get style text
                style = f'QCheckBox::indicator:{kw}'

                # Get style image
                imgPath = f'image: url({base_url}checkbox_{kw}'

                # Append additional info for 'hover'/'pressed' kws
                if subKw != '':
                    style = f'{style}:{subKw}'
                    imgPath = f'{imgPath}_{subKw}'

                # Merge style and image info into single string
                styleStr = f'{style} {{ {imgPath}.png);}}'

                if style == 'indeterminate' and subKw == '':
                    continue

                # Slash direction needs to be reversed in Windows paths for Qt for some reason
                if os.name == 'nt':
                    styleStr = styleStr.replace('\\', '/')

                checkBoxStyles.append(styleStr)

        checkBoxStyle = '\n'.join(checkBoxStyles)
        Frame.setStyleSheet(checkBoxStyle)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        self.clearButton = QtWidgets.QPushButton('Clear')
        self.removePlotButton = QtWidgets.QPushButton('Remove Plot')
        self.addPlotButton = QtWidgets.QPushButton('Add Plot')
        self.defaultsButton = QtWidgets.QPushButton('Defaults')
        self.switchButton = QtWidgets.QPushButton('Switch')
        self.plotButton = QtWidgets.QPushButton('Plot')

        self.errorFlagEdit = QtWidgets.QLineEdit('null')
        self.errorFlagEdit.setFixedWidth(50)

        errorFlagLayout = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel('    Error Flag:')
        lab.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        errorFlagLayout.addWidget(lab)
        errorFlagLayout.addWidget(self.errorFlagEdit)

        buttonLayout = QtWidgets.QGridLayout()
        topBtns = [self.clearButton, self.removePlotButton, self.addPlotButton,
            self.defaultsButton]
        btnNum = 0
        for btn in topBtns:
            buttonLayout.addWidget(btn, 0, btnNum, 1, 1)
            btnNum += 1

        buttonLayout.addWidget(self.plotButton, 1, 0, 1, 3)
        buttonLayout.addLayout(errorFlagLayout, 1, 4, 1, 1)

        self.editCombo = QtWidgets.QComboBox()
        buttonLayout.addWidget(self.editCombo, 1, 3, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        buttonLayout.addItem(spacer, 0, 10, 1, 1)
        self.layout.addLayout(buttonLayout)

        # Plot frame and outer grid layout that hold the data selection
        # layouts that may be dynamically set by the user
        self.pltLtFrame = QtWidgets.QFrame()
        self.pltLtContainer = QtWidgets.QGridLayout(self.pltLtFrame)
        self.plottingLayout = ListLayout(Frame.window, Frame)
        self.pltLtContainer.addLayout(self.plottingLayout, 0, 0, 1, 1)
        self.layout.addWidget(self.pltLtFrame)

        self.fgridFrame = QtWidgets.QGroupBox('Link Groups')
        self.fgridFrame.setToolTip('Link the Y axes of each plot in each group to have the same scale with each other')
        self.fgrid = QtWidgets.QGridLayout(self.fgridFrame)
        self.layout.addWidget(self.fgridFrame)

class PlotMenu(QtWidgets.QFrame, PlotMenuUI):
    def __init__(self, window, parent=None):
        super(PlotMenu, self).__init__(parent)

        self.window = window
        self.ui = PlotMenuUI()
        self.plotCount = 0
        self.tableMode = self.window.plotMenuTableMode
        self.ui.setupUI(self)
        self.ui.errorFlagEdit.setText(f'{self.window.errorFlag:.0e}')

        self.ABBRV_DSTRS = [self.window.ABBRV_DSTR_DICT[dstr] for dstr in self.window.DATASTRINGS]

        self.ui.clearButton.clicked.connect(self.clearRows)
        self.ui.addPlotButton.clicked.connect(self.addPlot)
        self.ui.removePlotButton.clicked.connect(self.removePlot)
        self.ui.plotButton.clicked.connect(self.plotData)
        self.ui.defaultsButton.clicked.connect(self.reloadDefaults)
        self.ui.switchButton.clicked.connect(self.switchModes)

        self.ui.switchButton.setText('Switch to Dropdown Lists' if self.tableMode else 'Switch to Table View')
        self.fcheckBoxes = []

        # add the edit name options to dropdown
        for editName in self.window.editNames:
            self.ui.editCombo.addItem(editName)
        self.ui.editCombo.currentTextChanged.connect(self.updtDstrOptions)

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks,
            self.window.lastPlotHeightFactors)

    def closeEvent(self, event):
        self.window.plotMenuTableMode = self.tableMode

    def updtDstrOptions(self):
        self.ui.plottingLayout.updtDstrOptions()

    def switchModes(self):
        # todo: save mode in magpy
        if self.ui.switchButton.text() == 'Switch to Table View':
            self.tableMode = True
            self.ui.switchButton.setText('Switch to Dropdown Lists')
        else:
            self.tableMode = False
            self.ui.switchButton.setText('Switch to Table View')

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks,
            self.window.lastPlotHeightFactors)
        self.ui.pltLtContainer.invalidate() #otherwise gridframe doesnt shrink back down
        QtCore.QTimer.singleShot(5, functools.partial(self.resize, 100, 100))

    def reloadDefaults(self):
        dstrs, links = self.window.getDefaultPlotInfo()
        self.ui.editCombo.setCurrentIndex(0)
        self.initPlotMenu(dstrs, links, [])

    def getLayout(self):
        layout = ListLayout(self.window, self)
        return layout

    def clearPreviousLayout(self):
        # Removes all elements from previous layout and deletes the layout
        self.ui.plottingLayout.clearElements()
        self.ui.pltLtContainer.removeItem(self.ui.plottingLayout)
        self.ui.plottingLayout.deleteLater()

    def initPlotMenu(self, dstrs, links, heightFactors):
        if dstrs is None:
            print('empty plot matrix!')
            return

        while self.ui.plottingLayout.plotCount() > 0:
            self.ui.plottingLayout.removePlot()
        self.ui.plottingLayout.initPlots(dstrs)

        self.plotCount = len(dstrs)

        if self.window.lastPlotLinks is not None:
            newLinks = []
            for l in links:
                if not isinstance(l, tuple):
                    newLinks.append(l)
            self.rebuildPlotLinks(len(newLinks))
            for i, axis in enumerate(newLinks):
                if i >= len(self.fcheckBoxes):
                    continue
                for j in axis:
                    if j >= len(self.fcheckBoxes[i]):
                        continue
                    self.fcheckBoxes[i][j].setChecked(True)

        # Fill in any height factors that aren't defined for plots w/ 
        # the default value of 1
        while len(heightFactors) < len(dstrs):
            heightFactors.append(1)

        for plotIndex in range(0, len(heightFactors)):
            factor = heightFactors[plotIndex]
            self.ui.plottingLayout.setHeightFactor(plotIndex, factor)

        width = 200
        height = 600
        if not self.isVisible() and len(dstrs) <= 4:
            height = 125 * len(dstrs) + 300

        if not self.isVisible():
            self.resize(width, height)

    # returns list of list of strings (one list for each plot, (dstr, editNumber) for each trace)
    def getPlotInfo(self):
        dstrs = self.ui.plottingLayout.getPltDstrs(1)
        return dstrs

    def getLinkLists(self):
        links = []
        if len(self.fcheckBoxes) == 0:
            return [(0,)]
        notFound = set([i for i in range(len(self.fcheckBoxes[0]))])
        for fAxis in self.fcheckBoxes:
            row = []
            for i,cb in enumerate(fAxis):
                if cb.isChecked():
                    row.append(i)
                    if i in notFound:
                        notFound.remove(i)
            if len(row) > 0:
                links.append(row)
        for i in notFound: # for anybody not part of set add them
            links.append([i])
        return links

    # callback for each checkbox on changed
    # which box, row and col are provided
    def checkPlotLinks(self, checkBox, r, c):
        if checkBox.isChecked(): # make sure ur only one checked in your column
            i = 0 # need to do it like this because setChecked callbacks can cause links to be rebuild mid iteration
            while i < len(self.fcheckBoxes):
                if i != r: # skip self
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
        return used if used < targ else targ

    # rebuild whole link table (tried adding and removing widgets and got pretty close but this was way easier in hindsight)
    # i think callbacks cause this to get called way more than it should but its pretty fast so meh
    def rebuildPlotLinks(self, targRows=None):
        fgrid = self.ui.fgrid

        self.targRows = self.getProperLinkRowCount()
        if targRows: # have at least targRows many
            self.targRows = max(self.targRows, targRows)

        cbools = self.checksToBools(self.fcheckBoxes, True)
        self.fcheckBoxes = [] # clear list after

        PyQtUtils.clearLayout(fgrid)

        # add top plot labels
        for i in range(self.plotCount):
            pLabel = QtWidgets.QLabel()
            pLabel.setText(f'Plot {i+1}')
            pLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            fgrid.addWidget(pLabel,0,i + 1,1,1)
        # add spacer
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        fgrid.addItem(spacer,0,self.plotCount + 1,1,1)

        # add checkBoxes
        for r in range(self.targRows):
            linkGroupLabel = QtWidgets.QLabel()
            linkGroupLabel.setText(f'Group {r+1}:')
            fgrid.addWidget(linkGroupLabel,r + 1,0,1,1)
            row = []
            for i in range(self.plotCount):
                checkBox = QtWidgets.QCheckBox()
                if r < len(cbools) and len(cbools) > 0 and i < len(cbools[0]):
                    checkBox.setChecked(cbools[r][i])
                # add callback with predefined arguments here
                checkBox.stateChanged.connect(functools.partial(self.checkPlotLinks, checkBox, r, i))
                row.append(checkBox)
                fgrid.addWidget(checkBox,r + 1,i + 1,1,1)      
            self.fcheckBoxes.append(row)

    def plotData(self):
        ''' Plots selected variables '''
        # try to update error flag
        try:
            flag = float(self.ui.errorFlagEdit.text())
            if self.window.errorFlag != flag:
                print(f'using new error flag: {flag:.0e}')
                self.window.errorFlag = flag
                self.window.reloadDataInterpolated()
        except ValueError:
            flag = self.window.errorFlag
            print(f'cannot interpret error flag value, leaving at: {flag:.0e}')
            self.ui.errorFlagEdit.setText(f'{flag:.0e}')

        # Get new plot data
        dstrs = self.getPlotInfo()
        links = self.getLinkLists()
        heightFactors = self.getHeightFactors()

        # Check that no more than one color plot is selected in
        # each plot
        for dstrLst in dstrs:
            count = 0
            for dstr, en in dstrLst:
                if en < 0:
                    count += 1
            if count > 1:
                msg = 'Error: Cannot have more than one spectrogram in a single plot'
                self.window.statusBar.showMessage(msg)
                return

        self.window.plotData(dstrs, links, heightFactors)

    def getHeightFactors(self):
        # Return empty list if in dropdown mode
        if not self.tableMode:
            return []

        # Extract the height factors from the plot layout sub-menus
        factors = self.ui.plottingLayout.getHeightFactors()
        return factors

    def clearRows(self):
        self.ui.plottingLayout.clearPlots()

    def addPlot(self):
        self.plotCount += 1
        self.ui.plottingLayout.addPlot()
        self.rebuildPlotLinks()

    def checksToBools(self, cbMatrix, skipEmpty=False):
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

    def checkIfPlotRemovable(self):
        if self.plotCount <= 1:
            self.window.ui.statusBar.showMessage('Error: Need at least one plot', 5000)
            return False
        return True

    def removePlot(self):
        if not self.checkIfPlotRemovable():
            return

        self.plotCount -= 1
        self.ui.plottingLayout.removePlot()
        self.rebuildPlotLinks()

    def getCurrentEdit(self):
        editNumber = max(0, self.ui.editCombo.currentIndex())
        return editNumber