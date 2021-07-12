
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg

import functools
from .MagPy4UI import PyQtUtils
from . import getRelPath
import os
import numpy as np
from .layoutTools import HBoxLayout, VBoxLayout

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

class DragLabel(QtWidgets.QLabel):
    ''' Drag handle widget facilitates drag events for the given widget '''
    def __init__(self, widget, *args, **kwargs):
        self.widget = widget
        QtWidgets.QLabel.__init__(self, *args, **kwargs)
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
        vertVis = self.verticalScrollBar().isVisible()
        horzVis = self.horizontalScrollBar().isVisible()
        if vertVis or horzVis:
            self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        else:
            self.setFrameShape(QtWidgets.QFrame.NoFrame)

class ListLayout(QtWidgets.QGridLayout):
    plotRemoved = QtCore.pyqtSignal(object)
    def __init__(self, window, frame):
        self.window = window
        self.mainFrame = frame
        self.editNumbers = []

        QtWidgets.QGridLayout.__init__(self)
        self.setupLayout()

    def setupLayout(self):
        # Sets up dstr selector and plot selections layout
        layout = self

        self.dstrTableFrame, self.dstrTable = self.setupDstrTable(layout)
        self.addSpecs()

        layout.addWidget(self.dstrTableFrame, 0, 0, 1, 1)

        self.plotWrappers = []
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

        # Notify other objects that plot was removed
        self.plotRemoved.emit(index)

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

    def addSpecs(self):
        for k in self.window.CDFSPECS:
            info = TraceInfo(k, -1, self.window)
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

    def getSelectedItems(self):
        return self.dstrTable.selectedItems()

    def getPlotWidgets(self):
        ''' Returns listwidgets for each plot '''
        count = self.pltLt.count()
        items = [self.pltLt.itemAtPosition(i, 0) for i in range(count)]
        widgets = [item.widget() for item in items]
        return widgets

    def splitAdd(self):
        ''' Distributes selected plot variables across first
            n plots, where n = len(plot variables)
        '''
        items = self.dstrTable.selectedItems()
        for item, plot in zip(items, self.getPlotWidgets()):
            plot.addItems([item])

class ButtonGroup(QtWidgets.QButtonGroup):
    ''' Button group that behaves similar to exclusive
        button group but allows all buttons to be unchecked
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setExclusive(False)
        self.buttonToggled.connect(self.adjustChecks)

    def adjustChecks(self, button, check):
        ''' If button is checked, clears checks
            from all other buttons
        '''
        if check:
            buttons = self.buttons()
            for other in buttons:
                if other != button:
                    other.setChecked(False)

class LinkBoxGroup(QtWidgets.QWidget):
    ''' A single row of checkboxes representing
        a link group
    '''
    def __init__(self, count=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = HBoxLayout(self)
        self.layout.setSpacing(16)
        self.layout.setContentsMargins(4, 2, 4, 4)
        self.setCount(count)

    def setCount(self, n):
        ''' Clears layout and adds n boxes '''
        self.layout.clear()
        for i in range(0, n):
            self.addColumn()

    def setChecked(self, indices):
        ''' Checks the buttons corresponding to given indices '''
        btns = self.getButtons()
        for i in indices:
            btns[i].setChecked(True)

    def getButtons(self):
        ''' Returns the buttons in the layout '''
        return self.layout.getItems()

    def getChecked(self):
        ''' Returns the indices of the selected boxes '''
        checked = []
        buttons = self.getButtons()
        for i in range(0, len(buttons)):
            if buttons[i].isChecked():
                checked.append(i)
        return checked

    def addColumn(self):
        ''' Add another box at the end '''
        box = QtWidgets.QCheckBox()
        self.layout.addWidget(box)
        return box

    def removeColumn(self, index=None):
        ''' Removes last box from layout '''
        self.layout.pop(index=index)

    def linkButtons(self, grps):
        ''' Adds buttons to respective button groups '''
        buttons = self.getButtons()
        for btn, grp in zip(buttons, grps):
            grp.addButton(btn)

class LinkPlotLabel(QtWidgets.QLabel):
    ''' Plot label item that also initializes a button
        group so that a plot may not be in more than one
        link group
    '''
    def __init__(self, index, *args, **kwargs):
        label = f'{index}'
        super().__init__(label, *args, **kwargs)
        self.group = ButtonGroup()

    def getButtonGroup(self):
        return self.group

class LinkWidget(QtWidgets.QGroupBox):
    def __init__(self, *args, **kwargs):
        super().__init__('Link Groups', *args, **kwargs)
        self.setupLayout()
        tt = 'Plots in the same group will have the same scale '''
        self.setToolTip(tt)

        self.addBtn.clicked.connect(self.addRow)
        self.minusBtn.clicked.connect(self.removeRow)

    def setupLayout(self):
        gridFrame = QtWidgets.QFrame()
        layout = QtWidgets.QGridLayout(gridFrame)
        layout.setVerticalSpacing(0)

        # Initialize sublayouts
        top_wrap = QtWidgets.QFrame()
        top_layout = HBoxLayout(top_wrap)

        left_wrap = QtWidgets.QFrame()
        left_layout = VBoxLayout(left_wrap)
        left_layout.setSpacing(0)

        grid_layout = VBoxLayout()
        grid_layout.setSpacing(0)

        # Set margins and size policies
        for lt in [top_layout, left_layout]:
            lt.setContentsMargins(0, 0, 0, 4)
        maxsp = QSizePolicy.Maximum
        prefsp = QSizePolicy.Preferred
        left_wrap.setSizePolicy(maxsp, prefsp)

        # Spacer item at end
        expsp = QSizePolicy.Expanding
        spacer1 = QtWidgets.QSpacerItem(0, 0, expsp, maxsp)
        spacer2 = QtWidgets.QSpacerItem(0, 0, prefsp, expsp)

        # Plot label label
        pltLbl = QtWidgets.QLabel('Plot #:')
        pltLbl.setAlignment(QtCore.Qt.AlignRight)
        layout.setAlignment(pltLbl, QtCore.Qt.AlignBaseline)

        # Add to main layout
        layout.addWidget(pltLbl, 0, 0, 1, 1)
        layout.addWidget(top_wrap, 0, 1, 1, 1)
        layout.addWidget(left_wrap, 1, 0, 1, 1)
        layout.addLayout(grid_layout, 1, 1, 1, 1)
        layout.addItem(spacer1, 0, 2, 2, 1)
        layout.addItem(spacer2, 2, 0, 1, 2)

        # Create wrapper layout and add buttons and
        # main inner layout
        self.addBtn = QtWidgets.QPushButton('+')
        self.minusBtn = QtWidgets.QPushButton('-')
        btnLt = QtWidgets.QVBoxLayout()
        btnLt.setSpacing(2)
        for btn in [self.addBtn, self.minusBtn]:
            btnLt.addWidget(btn)
            btn.setMaximumWidth(50)

        # Store sublayouts
        self.grp_lbl_lt = left_layout
        self.grid = grid_layout
        self.plt_lbl_lt = top_layout

        # Create scroll area and add grid frame to it
        self.scrollArea = AdjustingScrollArea()
        self.scrollArea.setWidget(gridFrame)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Add scroll area and btn layout to wrapper layout
        wrapLt = QtWidgets.QGridLayout(self)
        wrapLt.setContentsMargins(0, 0, 5, 0)
        wrapLt.addWidget(self.scrollArea, 0, 0, 1, 1)
        wrapLt.addLayout(btnLt, 0, 1, 1, 1)

    def setLinks(self, links, numPlots):
        ''' Creates and sets link group checkboxes
            based on given links and number of plots
        '''
        # Clear and initialize grid
        self.setShape(len(links), numPlots)

        # Iterate over link groups and set checkboxes
        # for each one
        widgets = self.getLinkWidgets()
        for i in range(0, len(links)):
            row_links = links[i]
            widget = widgets[i]
            widget.setChecked(row_links)

    def clear(self):
        ''' Clears all sublayouts'''
        self.grp_lbl_lt.clear()
        self.grid.clear()
        self.plt_lbl_lt.clear()

    def numCols(self):
        return self.plt_lbl_lt.count()

    def numRows(self):
        return self.grp_lbl_lt.count()

    def getLinkWidgets(self):
        ''' Returns the LinkBoxGroup widgets in the grid '''
        widgets = self.grid.getItems()
        return widgets

    def setShape(self, r, c):
        ''' Clears grid and sets up a grid of checkboxes
            with r rows and c columns '''
        self.clear()
        for i in range(0, c):
            self.addColumn()

        for j in range(0, r):
            self.addRow()

    def addRow(self):
        ''' Adds another row of checkboxes and a group label
            to the grid
        '''
        # Add another group label on left grid
        rows = self.numRows()
        label = f'Group {rows+1}:'
        label = QtWidgets.QLabel(label)
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.grp_lbl_lt.addWidget(label)

        # Add another link widget
        cols = self.numCols()
        widget = LinkBoxGroup(cols)
        self.grid.addWidget(widget)

        # Add new row of buttons to button groups
        btn_grps = self._getButtonGroups()
        widget.linkButtons(btn_grps)

        # Update size if necessary
        self._updateSize()

    def addColumn(self):
        ''' Adds another plot column to the grid '''
        # Add another plot label on top grid
        cols = self.numCols()
        label = LinkPlotLabel(cols+1)
        self.plt_lbl_lt.addWidget(label)

        # Align label in center
        align = QtCore.Qt.AlignHCenter
        self.plt_lbl_lt.setAlignment(label, align)

        # Add a button to each link group and add it to
        # the button group for the current plot
        btn_grp = label.getButtonGroup()
        widgets = self.getLinkWidgets()
        for widget in widgets:
            box = widget.addColumn()
            btn_grp.addButton(box)

        # Hide if there is only one plot
        if cols == 1:
            self.setVisible(True)

    def removeColumn(self, col=None):
        ''' Removes last (or given col #) plot column from grid '''
        # Remove last column from each link group
        widgets = self.getLinkWidgets()
        for widget in widgets:
            widget.removeColumn(col)

        # Remove last plot label
        self.plt_lbl_lt.pop()

        # Show if previously hidden
        if self.numCols() == 1:
            self.setVisible(False)

    def removeRow(self):
        ''' Removes last row of checkboxes / link group from grid '''
        # Remove last row from group labels and button grid
        if self.grid.count() > 1:
            self.grp_lbl_lt.pop()
            self.grid.pop()

        # Update size if necessary
        self._updateSize()

    def getLinks(self):
        ''' Returns the groups of plots to link together
            (skipping any empty groups)
        '''
        widgets = self.getLinkWidgets()
        links = []
        for widget in widgets:
            group = widget.getChecked()
            if len(group) > 0:
                links.append(group)

        return links

    def _getButtonGroups(self):
        ''' Returns the button groups associated with each plot '''
        labels = self.plt_lbl_lt.getItems()
        groups = [label.getButtonGroup() for label in labels]
        return groups

    def _updateSize(self):
        ''' Updates minimum scroll area height '''
        n = min(self.numRows(), 3)
        sizes = {0: 75, 1:75, 2:110, 3:140}
        size = sizes[n]
        self.scrollArea.setMinimumHeight(size)

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

        self.layout = QtWidgets.QVBoxLayout(Frame)

        self.clearButton = QtWidgets.QPushButton('Clear')
        self.removePlotButton = QtWidgets.QPushButton('Remove Plot')
        self.addPlotButton = QtWidgets.QPushButton('Add Plot')
        self.defaultsButton = QtWidgets.QPushButton('Defaults')
        self.saveStringsBtn = QtWidgets.QCheckBox('Keep Plot Choices')
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
        buttonLayout.addWidget(self.saveStringsBtn, 0, 4, 1, 1)
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
        self.pltLtContainer.setContentsMargins(0, 10, 0, 0)
        self.plottingLayout = ListLayout(Frame.window, Frame)
        self.pltLtContainer.addLayout(self.plottingLayout, 0, 0, 1, 1)
        self.layout.addWidget(self.pltLtFrame)

        # Set up split button
        img_path = getRelPath('images')
        img_path = os.path.join(img_path, 'split_arrow.png')
        pixmap = QtGui.QPixmap(img_path)
        icon = QtGui.QIcon(pixmap)

        self.splitAddBtn = QtWidgets.QPushButton('')
        self.splitAddBtn.setIcon(icon)
        self.splitAddBtn.setIconSize(QtCore.QSize(25, 25))
        tt = 'Click to distribute selected N variables across first N plots'
        self.splitAddBtn.setToolTip(tt)

        splitBtnLt = QtWidgets.QHBoxLayout()
        splitBtnLt.addStretch(1)
        splitBtnLt.addWidget(self.splitAddBtn)
        splitBtnLt.addStretch(1)
        splitBtnLt.setContentsMargins(0, 0, 0, 0)

        self.layout.addLayout(splitBtnLt)

        # Set up link button layout
        self.fgridFrame = QtWidgets.QGroupBox('Link Groups')
        self.fgridFrame.setToolTip('Link the Y axes of each plot in each group to have the same scale with each other')
        self.fgrid = LinkWidget()
        self.layout.addWidget(self.fgrid)
        self.fgrid.setStyleSheet(checkBoxStyle)

        # Connect plot removed menu actions to link buttons
        self.plottingLayout.plotRemoved.connect(self.fgrid.removeColumn)
        self.plottingLayout.plotRemoved.connect(Frame.decrementPlotCount)

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
        self.ui.splitAddBtn.clicked.connect(self.splitAdd)
        self.fcheckBoxes = []

        # add the edit name options to dropdown
        for editName in self.window.editNames:
            self.ui.editCombo.addItem(editName)
        self.ui.editCombo.currentTextChanged.connect(self.updtDstrOptions)

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks,
            self.window.lastPlotHeightFactors)

        # Check keep plot choices if there is saved plot info
        if self.window.savedPlotInfo is not None:
            self.ui.saveStringsBtn.setChecked(True)

        # Save plot choices if toggled
        self.ui.saveStringsBtn.toggled.connect(self.savePlotChoices)

    def savePlotChoices(self, val):
        if val:
            dstrs = self.window.lastPlotStrings
            links = self.window.lastPlotLinks
            heights = self.window.lastPlotHeightFactors
            self.window.savedPlotInfo = (dstrs, links, heights)
        else:
            self.window.savedPlotInfo = None

    def closeEvent(self, event):
        self.window.plotMenuTableMode = self.tableMode

    def updtDstrOptions(self):
        self.ui.plottingLayout.updtDstrOptions()

    def switchModes(self):
        # todo: save mode in magpy
        self.tableMode = True

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
            self.ui.fgrid.setLinks(self.window.lastPlotLinks, len(dstrs))

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

    def getLinks(self):
        ''' Returns the plot link groups '''
        # Get plot links from grid
        links = self.ui.fgrid.getLinks()

        # Find out which plots have been placed in a link group
        seenPlots = set()
        for link in links:
            seenPlots = seenPlots | set(link)

        # Get a list of all plot indices without a link group
        fullset = set([i for i in range(self.plotCount)])
        unseen = fullset - seenPlots

        # Add unseen plot indices to their own link groups
        for i in unseen:
            links.append([i])

        return links

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
        links = self.getLinks()
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

        save = self.ui.saveStringsBtn.isChecked()

        self.window.plotData(dstrs, links, heightFactors, save=save)

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
        self.ui.fgrid.addColumn()

    def decrementPlotCount(self):
        self.plotCount -= 1

    def splitAdd(self):
        items = self.ui.plottingLayout.getSelectedItems()
        for i in range(self.plotCount, len(items)):
            self.addPlot()

        self.ui.plottingLayout.splitAdd()

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
        self.ui.fgrid.removeColumn()

    def getCurrentEdit(self):
        editNumber = max(0, self.ui.editCombo.currentIndex())
        return editNumber