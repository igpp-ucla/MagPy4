
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import functools
from .MagPy4UI import PyQtUtils
from . import getRelPath
import os

class AdjustingScrollArea(QtWidgets.QScrollArea):
    # Scroll area that hides border when vertical scrollbar is not visible
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum))

    def resizeEvent(self, ev):
        QtWidgets.QScrollArea.resizeEvent(self, ev)
        self.updateFrameBorder()

    def updateFrameBorder(self):
        if self.verticalScrollBar().isVisible():
            self.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.setStyleSheet('QScrollArea { background-color : #ffffff; }')
        else:
            self.setFrameShape(QtWidgets.QFrame.NoFrame)

class DropdownLayout(QtWidgets.QGridLayout):
    def __init__(self, window, frame):
        self.grid = self
        self.window = window
        self.mainFrame = frame
        self.dropdowns = []
        self.plotLabels = []

        QtWidgets.QGridLayout.__init__(self)

    def addPlot(self):
        en = self.mainFrame.getCurrentEdit()

        # Adds the plot label and empty boxes filled w/ current edit's dstrs
        pltNum = len(self.dropdowns)
        lbl = QtWidgets.QLabel('Plot ' + str(pltNum+1) + ':')
        self.grid.addWidget(lbl, pltNum, 0, 1, 1)
        dd = self.addEmptyBox(pltNum, 1)

        # Store dropdown and labels for removing later
        self.dropdowns.append([(dd, en)])
        self.plotLabels.append(lbl)

    def removePlot(self):
        # Remove dropdowns and plot label from layout and delete
        dropRow = self.dropdowns[-1]
        for dd, en in dropRow:
            self.grid.removeWidget(dd)
            dd.deleteLater()

        lbl = self.plotLabels[-1]
        self.grid.removeWidget(lbl)
        lbl.deleteLater()

        # Update lists that keep track of dropdowns
        for lst in [self.dropdowns, self.plotLabels]:
            lst.pop()

    def initPlots(self, dropList):
        self.checkBoxes = []
        # for each dropdown row in all the dropdowns
        for di, dropRow in enumerate(dropList):
            # add spacer in first row to take up right space
            if di == 0:
                spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
                self.grid.addItem(spacer,0,100,1,1) # just so its always the last column

            pltLbl = QtWidgets.QLabel('Plot ' + str(di+1) + ':')
            self.grid.addWidget(pltLbl, di, 0, 1, 1)
            self.plotLabels.append(pltLbl)

            dropRow = dropRow + [('', self.mainFrame.getCurrentEdit())]

            row = []
            # for each dropdown in row
            for i,(dstr,en) in enumerate(dropRow):
                if en < 0:
                    dd = QtWidgets.QComboBox()
                    dd.addItem(dstr)
                    dd.addItem('')
                    self.grid.addWidget(dd, di, i+1, 1, 1)
                    dd.currentTextChanged.connect(self.resetDropdownLsts)
                elif dstr != '':
                    dd = self.addBox(di, i+1, en, dstr)
                else:
                    dd = self.addEmptyBox(di, i+1)
                row.append((dd,en))

            self.dropdowns.append(row)

    def clearPlots(self):
        # Remove all plots and re-add the same number as before
        numPlts = len(self.dropdowns)
        self.clearElements()
        for i in range(0, numPlts):
            self.addPlot()

    def resetDropdownLsts(self):
        # Re-organizes plot dropdowns if an empty box is filled or a previously
        # filled box is emptied
        rowNum = 0
        prevDropdowns = self.dropdowns[:]
        self.dropdowns = []
        for ddRow in prevDropdowns:
            # Find the empty and non-empty combo boxes in this row
            nonEmptyBoxes = []
            for dd, en in ddRow:
                dstr = dd.currentText()
                # Remove all dropdowns from this row
                self.grid.removeWidget(dd) 
                if dstr == '':
                    dd.deleteLater()
                else:
                    nonEmptyBoxes.append((dd, en))

            # Add back in all non-empty boxes
            col = 1
            for dd, en in nonEmptyBoxes:
                self.grid.addWidget(dd, rowNum, col, 1, 1)
                col += 1

            # Add an empty box at the end
            emptyEndBx = self.addEmptyBox(rowNum, col)

            # Save the new list of dropdowns
            nonEmptyBoxes.append((emptyEndBx, self.mainFrame.getCurrentEdit()))
            self.dropdowns.append(nonEmptyBoxes)

            rowNum += 1

    def addBox(self, row, col, en, dstr):
        # Creates an empty box first
        box = QtWidgets.QComboBox()
        self.fillComboBox(box, en)
        self.grid.addWidget(box, row, col, 1, 1)

        # Finds the index of the dstr it's to be set to
        ddItems = [box.itemText(itemIndex) for itemIndex in range(0, box.count())]
        currIndex = ddItems.index(self.window.getLabel(dstr, en))
        box.setCurrentIndex(currIndex)

        box.currentTextChanged.connect(self.resetDropdownLsts)
        return box

    def addEmptyBox(self, row, col):
        # Adds an empty box to grid and fills it
        emptyEndBx = QtWidgets.QComboBox()
        self.fillComboBox(emptyEndBx, self.mainFrame.getCurrentEdit())
        self.grid.addWidget(emptyEndBx, row, col, 1, 1)
        emptyEndBx.currentTextChanged.connect(self.resetDropdownLsts)
        return emptyEndBx

    def fillComboBox(self, dd, en):
        # Fills the combo box w/ dstrs for the given edit number
        dd.addItem('')
        for k,v in self.window.DATADICT.items():
            if len(v[en]) > 0:
                s = self.window.getLabel(k,en)
                dd.addItem(s)

    def updtDstrOptions(self):
        self.resetDropdownLsts()

    def getPltDstrs(self, pltNum):
        ddRow = self.dropdowns[pltNum]
        dstrs = []
        for dd, en in ddRow:
            dstr = dd.currentText()
            if dstr not in dstrs:
                dstrs.append(dstr)
        return dstrs

    def clearElements(self):
        numPlts = len(self.dropdowns)
        while numPlts > 0:
            self.removePlot()
            numPlts -= 1

    def setHeightFactor(self, plotItem, factor):
        return

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

        self.pltFrms, self.pltTbls, self.elems = [], [], []
        self.pltLtFrm = QtWidgets.QFrame()
        self.pltLt = QtWidgets.QGridLayout(self.pltLtFrm)

        # Set up scroll area wrapper for plot boxes
        self.scrollArea = AdjustingScrollArea()
        self.scrollArea.setWidget(self.pltLtFrm)
        layout.addWidget(self.scrollArea, 0, 1, 1, 1)

    def setupDstrTable(self, layout):
        frame = QtWidgets.QGroupBox('Data Variables')
        layout = QtWidgets.QVBoxLayout(frame)
        table = QtWidgets.QListWidget()

        table.insertItems(0, self.window.DATASTRINGS)
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

        # Create layout
        pltElems = [pltLbl, optBtn]
        for elem in pltElems:
            pltTitleLt.addWidget(elem)

        return pltTitleLt, pltElems + [heightBox]

    def closeSubPlot(self, plotIndex):
        if not self.mainFrame.checkIfPlotRemovable():
            return

        # Remove plot, update frame state, and update plot titles
        self.removeSpecificPlot(plotIndex)
        self.updatePlotTitles()
        self.mainFrame.plotCount -= 1
        self.mainFrame.rebuildPlotLinks()

    def addPlot(self):
        plotNum = len(self.pltFrms) + 1
        row = plotNum - 1
        col = 0

        plotTableLt = QtWidgets.QGridLayout()
        self.pltLt.addLayout(plotTableLt, row, col, 1, 1)

        # Add plot label
        pltTitleLt, pltTitleElems = self.getPlotTitleLt(plotNum)
        plotTableLt.addLayout(pltTitleLt, 0, 1, 1, 1)

        # Add plot table
        table = QtWidgets.QListWidget()
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        plotTableLt.addWidget(table, 1, 1, 1, 1)

        # Set up buttons from adding/removing dstrs from plots
        pltBtnLt = QtWidgets.QVBoxLayout()
        plotTableLt.addLayout(pltBtnLt, 1, 0, 1, 1)

        pltBtn = QtWidgets.QPushButton('>>')
        pltRmvBtn = QtWidgets.QPushButton(' < ')

        pltBtnLt.addStretch()
        for btn in [pltBtn, pltRmvBtn]:
            btn.setFixedWidth(50)
            pltBtnLt.addWidget(btn)
        pltBtnLt.addStretch()

        # Store this layout and elems for removing + the plot table itself
        self.pltFrms.append(plotTableLt)
        self.pltTbls.append(table)
        elemList = [pltBtn, pltRmvBtn] + pltTitleElems
        self.elems.append(elemList)

        # Connect add/remove buttons to functions
        pltBtn.clicked.connect(functools.partial(self.subPlotAction, pltBtn))
        pltRmvBtn.clicked.connect(functools.partial(self.subPlotAction, pltRmvBtn))
        return plotTableLt, table

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

    def subMenuAction(self, btn, actType):
        # Find the index of the menu's frame first
        index = 0
        found = False
        for addBtn, rmvBtn, lbl, optBtn, heightBox in self.elems:
            if optBtn == btn:
                found = True
                break
            index += 1

        # Apply action if a matching plot is found
        if found:
            if actType == 'Remove':
                self.closeSubPlot(index)
            elif actType == 'Up':
                self.movePlotUp(index)
            elif actType == 'Down':
                self.movePlotDown(index)

    def setHeightFactor(self, plotIndex, factor):
        box = self.elems[plotIndex][4]
        box.setValue(factor)

    def switchTables(self, plotIndex_1, plotIndex_2):
        # Switch the set of items in each table with the items from the other
        # table
        upperTable = self.pltTbls[plotIndex_1]
        lowerTable = self.pltTbls[plotIndex_2]

        upperDstrs = [upperTable.item(row) for row in range(upperTable.count())]
        upperDstrs = [item.text() for item in upperDstrs]
        lowerDstrs = [lowerTable.item(row) for row in range(lowerTable.count())]
        lowerDstrs = [item.text() for item in lowerDstrs]

        upperTable.clear()
        lowerTable.clear()

        for item in upperDstrs:
            lowerTable.addItem(item)

        for item in lowerDstrs:
            upperTable.addItem(item)

        # Switch height factor settings
        upperBox = self.elems[plotIndex_1][4]
        upperHeight = upperBox.value()
        lowerBox = self.elems[plotIndex_2][4]
        lowerHeight = lowerBox.value()

        upperBox.setValue(lowerHeight)
        lowerBox.setValue(upperHeight)

    def movePlotUp(self, index):
        if index <= 0:
            return

        # Switch list items w/ list below current index
        self.switchTables(index-1, index)

    def movePlotDown(self, index):
        if index >= len(self.pltTbls) - 1:
            return

        # Switch list items w/ list above current index
        self.switchTables(index, index+1)

    def removePlot(self):
        # Remove plot layout from layout and delete the plot elements
        self.pltLt.removeItem(self.pltFrms[-1])
        self.pltTbls[-1].deleteLater()
        for elem in self.elems[-1]:
            elem.deleteLater()

        # Update lists that keep track of plot tables and elements
        for lst in [self.pltFrms, self.pltTbls, self.elems]:
            lst.pop()

    def removeSpecificPlot(self, index):
        # Remove plot layout from layout and delete the plot elements
        self.pltLt.removeItem(self.pltFrms[index])
        self.pltTbls[index].deleteLater()

        for elem in self.elems[index]:
            elem.deleteLater()

        # Update lists that keep track of plot tables and elements
        for lst in [self.pltFrms, self.pltTbls, self.elems]:
            lst.pop(index)

        # Shift all plot layouts above this index down by one index
        for row in range(index, len(self.pltFrms)):
            lt = self.pltFrms[row]
            self.pltLt.removeItem(lt)
            self.pltLt.addLayout(lt, row, 0, 1, 1)

    def updatePlotTitles(self):
        plotNum = 1
        for elemLst in self.elems:
            pltLbl = elemLst[2]
            pltLbl.setText('Plot ' + str(plotNum) + ':') 
            plotNum += 1

    def initPlots(self, dstrLst):
        # Creates a table for each dstr sub-list and fills it w/ the given dstrs
        pltNum = 0
        for subLst in dstrLst:
            self.addPlot()
            dstrs = []
            for (dstr, en) in subLst:
                if en < 0:
                    dstrs.append(dstr)
                else:
                    lbl = self.window.getLabel(dstr, en)
                    dstrs.append(lbl)
            self.addSpecificDstrsToPlt(len(self.pltTbls)-1, dstrs)
            pltNum += 1

    def clearPlots(self):
        for plt in self.pltTbls:
            plt.clear()

    def updtDstrOptions(self):
        # Update dstr table w/ current edit's dstrs
        self.dstrTable.clear()
        editNum = self.mainFrame.getCurrentEdit()
        newDstrs = []
        for k,v in self.window.DATADICT.items():
            if len(v[editNum]) > 0:
                s = self.window.getLabel(k,editNum)
                newDstrs.append(s)
        self.dstrTable.addItems(newDstrs)

    def rmvDstrsFrmPlt(self, plotNum):
        # Removes the selected datastrings from the given plot table
        selectedItems = self.pltTbls[plotNum].selectedItems()

        for item in selectedItems:
            row = self.pltTbls[plotNum].row(item)
            self.pltTbls[plotNum].takeItem(row)

    def addDstrsToPlt(self, plotNum):
        # Adds the selected dstrs from the dstr table to the given plot table
        selectedItems = self.dstrTable.selectedItems()
        selectedDstrs = [item.text() for item in selectedItems]

        self.addSpecificDstrsToPlt(plotNum, selectedDstrs)

    def addSpecificDstrsToPlt(self, plotNum, selectedDstrs):
        # Used by addDstrsToPlt and initPlots to update a plot table w/ dstrs
        prevDstrs = self.getPltDstrs(plotNum)
        for dstr in selectedDstrs:
            if dstr not in prevDstrs:
                self.pltTbls[plotNum].addItem(dstr)

    def getPltDstrs(self, pltNum):
        pltList = self.pltTbls[pltNum]
        n = pltList.count()
        dstrs = []
        for i in range(0, n):
            dstrs.append(pltList.item(i).text())
        return dstrs

    def clearElements(self):
        numPlts = len(self.pltFrms)
        if numPlts == 0:
            return
        # Removes all plots, then removes the dstr table/frame
        while numPlts > 0:
            self.removePlot()
            numPlts -= 1
        self.removeWidget(self.dstrTable)
        self.removeWidget(self.dstrTableFrame)
        self.dstrTable.deleteLater()
        self.dstrTableFrame.deleteLater()
        self.removeWidget(self.scrollArea)
        self.scrollArea.deleteLater()

class PlotMenuUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Plot Menu')
        Frame.resize(100,100)

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
            self.defaultsButton, self.switchButton]
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
        self.plottingLayout = DropdownLayout(Frame.window, Frame)
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
        if self.tableMode:
            layout = ListLayout(self.window, self)
        else:
            layout = DropdownLayout(self.window, self)
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

        self.clearPreviousLayout()
        self.ui.plottingLayout = self.getLayout()
        self.ui.pltLtContainer.addLayout(self.ui.plottingLayout, 0, 0, 1, 1)
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

    # returns list of list of strings (one list for each plot, (dstr, editNumber) for each trace)
    def getPlotInfo(self):
        dstrs = []
        for i in range(0, self.plotCount):
            # Gets the list of selected datastrings from the layout
            pltDstrs = self.ui.plottingLayout.getPltDstrs(i)

            # Then it looks through all combinations of edit numbers and
            # dstrs to find the dstr w/ the matching edit name
            dstrWithEditNums = []
            for dstr in pltDstrs:
                if dstr == '':
                    continue
                elif dstr in self.window.pltGrd.colorPltKws:
                    dstrWithEditNums.append((dstr, -1))
                    continue
                for k, v in self.window.DATADICT.items():
                    for en in range(0, len(v)):
                        editStr = self.window.getLabel(k, en)
                        if (editStr == dstr and len(v[en]) > 0):
                            dstrWithEditNums.append((k, en))
                            break
            dstrs.append(dstrWithEditNums)
        if dstrs == []:
            dstrs.append([])
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

        dstrs = self.getPlotInfo()
        links = self.getLinkLists()
        heightFactors = self.getHeightFactors()
        self.window.plotData(dstrs, links, heightFactors)

    def getHeightFactors(self):
        # Return empty list if in dropdown mode
        if not self.tableMode:
            return []

        # Extract the height factors from the plot layout sub-menus
        factors = []
        for elems in self.ui.plottingLayout.elems:
            box = elems[4]
            factors.append(box.value())
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