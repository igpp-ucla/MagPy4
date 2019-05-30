
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import functools
from MagPy4UI import PyQtUtils

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
        self.pltLt = QtWidgets.QGridLayout()

        layout.addLayout(self.pltLt, 0, 1, 1, 1)

    def setupDstrTable(self, layout):
        frame = QtWidgets.QGroupBox('Data Variables')
        layout = QtWidgets.QVBoxLayout(frame)
        table = QtWidgets.QListWidget()

        table.insertItems(0, self.window.DATASTRINGS)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        layout.addWidget(table)
        return frame, table

    def addPlot(self):
        plotNum = len(self.pltFrms) + 1

        # Limit the number of plots per column
        if (plotNum - 1) > 5:
            row, col = (plotNum - 1) % 6, int((plotNum - 1)/6) + 1
        else:
            row, col = plotNum - 1, 1

        plotTableLt = QtWidgets.QGridLayout()
        self.pltLt.addLayout(plotTableLt, row, col, 1, 1)

        # Add plot label
        pltLbl = QtWidgets.QLabel('Plot ' + str(plotNum) + ':')
        plotTableLt.addWidget(pltLbl, 0, 1, 1, 1)

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
        self.elems.append([pltLbl, pltBtn, pltRmvBtn])

        # Connect add/remove buttons to functions
        pltBtn.clicked.connect(functools.partial(self.addDstrsToPlt, plotNum-1))
        pltRmvBtn.clicked.connect(functools.partial(self.rmvDstrsFrmPlt, plotNum-1))
        return plotTableLt, table

    def removePlot(self):
        # Remove plot layout from layout and delete the plot elements
        self.pltLt.removeItem(self.pltFrms[-1])
        self.pltTbls[-1].deleteLater()
        for elem in self.elems[-1]:
            elem.deleteLater()

        # Update lists that keep track of plot tables and elements
        for lst in [self.pltFrms, self.pltTbls, self.elems]:
            lst.pop()

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

class PlotMenuUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Plot Menu')
        Frame.resize(100,100)

        checkBoxStyle = """
            QCheckBox{spacing: 0px;}
            QCheckBox::indicator{width:32px;height:32px}
            QCheckBox::indicator:unchecked {            image: url(images/checkbox_unchecked.png);}
            QCheckBox::indicator:unchecked:hover {      image: url(images/checkbox_unchecked_hover.png);}
            QCheckBox::indicator:unchecked:pressed {    image: url(images/checkbox_unchecked_pressed.png);}
            QCheckBox::indicator:checked {              image: url(images/checkbox_checked.png);}
            QCheckBox::indicator:checked:hover {        image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:checked:pressed {      image: url(images/checkbox_checked_pressed.png);}
            QCheckBox::indicator:indeterminate:hover {  image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:indeterminate:pressed {image: url(images/checkbox_checked_pressed.png);}
        """

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

        self.fgridFrame = QtWidgets.QGroupBox('Y Axis Link Groups')
        self.fgridFrame.setToolTip('Link the Y axes of each plot in each group to have the same scale with each other')
        self.fgrid = QtWidgets.QGridLayout(self.fgridFrame)
        self.layout.addWidget(self.fgridFrame)

        # make invisible stretch to take up rest of space
        self.layout.addStretch()

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

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks)

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

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks)
        self.ui.pltLtContainer.invalidate() #otherwise gridframe doesnt shrink back down
        QtCore.QTimer.singleShot(5, functools.partial(self.resize, 100, 100))

    def reloadDefaults(self):
        dstrs, links = self.window.getDefaultPlotInfo()
        self.ui.editCombo.setCurrentIndex(0)
        self.initPlotMenu(dstrs, links)

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

    def initPlotMenu(self, dstrs, links):
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
                elif dstr in self.window.pltGrd.colorPltNames:
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

        self.window.plotData(dstrs, links)
        self.window.pltGrd.resizeEvent(None)

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

    def removePlot(self):
        if self.plotCount <= 1:
            print('need at least one plot')
            return

        self.plotCount-=1
        self.ui.plottingLayout.removePlot()
        self.rebuildPlotLinks()

        # Update window size
        self.ui.pltLtContainer.invalidate() #otherwise gridframe doesnt shrink back down
        if self.plotCount < 3:
            QtCore.QTimer.singleShot(5, functools.partial(self.resize, 100, 100))

    def getCurrentEdit(self):
        editNumber = max(0, self.ui.editCombo.currentIndex())
        return editNumber