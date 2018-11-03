
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import functools
from MagPy4UI import PyQtUtils

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
        lab = QtWidgets.QLabel('  Error Flag')
        lab.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        errorFlagLayout.addWidget(lab)
        errorFlagLayout.addWidget(self.errorFlagEdit)
        
        buttonLayout = QtWidgets.QGridLayout()
        buttonLayout.addWidget(self.clearButton, 0, 0, 1, 1)
        buttonLayout.addWidget(self.removePlotButton, 0, 1, 1, 1)
        buttonLayout.addWidget(self.addPlotButton, 0, 2, 1, 1)
        buttonLayout.addWidget(self.defaultsButton, 0, 3, 1, 1)
        buttonLayout.addWidget(self.switchButton, 0, 4, 1, 1)

        buttonLayout.addWidget(self.plotButton, 1, 0, 1, 3)
        buttonLayout.addLayout(errorFlagLayout, 1, 3, 1, 1)

        self.editCombo = QtWidgets.QComboBox()
        buttonLayout.addWidget(self.editCombo, 1, 4, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        buttonLayout.addItem(spacer, 0, 10, 1, 1)

        self.layout.addLayout(buttonLayout)

        self.gridFrame = QtWidgets.QGroupBox('Plot Matrix')
        self.grid = QtWidgets.QGridLayout(self.gridFrame)
        self.layout.addWidget(self.gridFrame)

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
        self.ui.setupUI(self)
        self.ui.errorFlagEdit.setText(f'{self.window.errorFlag:.0e}')

        self.ABBRV_DSTRS = [self.window.ABBRV_DSTR_DICT[dstr] for dstr in self.window.DATASTRINGS]

        self.ui.clearButton.clicked.connect(self.clearRows)
        self.ui.addPlotButton.clicked.connect(self.addPlot)
        self.ui.removePlotButton.clicked.connect(self.removePlot)
        self.ui.plotButton.clicked.connect(self.plotData)
        self.ui.defaultsButton.clicked.connect(self.reloadDefaults)
        self.ui.switchButton.clicked.connect(self.switchModes)

        self.checkBoxMode = self.window.plotMenuCheckBoxMode
        self.ui.switchButton.setText('Switch to Dropdowns' if self.checkBoxMode else 'Switch to CheckBoxes')
        self.fcheckBoxes = []

        # add the edit name options to dropdown
        for editName in self.window.editNames:
            self.ui.editCombo.addItem(editName)
        self.ui.editCombo.currentTextChanged.connect(self.rebuildDropdowns)

        self.shouldResizeWindow = False # gets set when switching modes
        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks)


    def closeEvent(self, event):
        self.window.plotMenuCheckBoxMode = self.checkBoxMode

    def paintEvent(self, event):
        if self.shouldResizeWindow:
            self.shouldResizeWindow = False
            self.resize(self.ui.layout.sizeHint())

    def switchModes(self):
        # todo: save mode in magpy
        if self.ui.switchButton.text() == 'Switch to CheckBoxes':
            self.checkBoxMode = True
            self.ui.switchButton.setText('Switch to Dropdowns')
        else:
            self.checkBoxMode = False
            self.ui.switchButton.setText('Switch to CheckBoxes')

        self.initPlotMenu(self.window.lastPlotStrings, self.window.lastPlotLinks)
        self.shouldResizeWindow = True
        
    def reloadDefaults(self):
        dstrs, links = self.window.getDefaultPlotInfo()
        self.initPlotMenu(dstrs, links)

    def initPlotMenu(self, dstrs, links):
        self.plotCount = 0
        self.checkBoxes = []
        self.dropdowns = [] # list of lists for each row and each element in row is (combobox, editNum)
        self.dropdownLocked = [] # list of sets for each plot of locked options
        PyQtUtils.clearLayout(self.ui.grid)

        if dstrs is None:
            print('empty plot matrix!')
            return

        if self.checkBoxMode:
            self.addLabels()
            for i,dstrList in enumerate(dstrs):
                self.addPlot()
                for dstr,editNum in dstrList:
                    di = self.window.DATASTRINGS.index(dstr)
                    self.checkBoxes[i][di].setChecked(True)
        else:
            for i,dstrList in enumerate(dstrs):
                self.plotCount += 1

            adstrs = []
            for dstrList in dstrs:
                adstrs.append([(dstr,editNum) for dstr,editNum in dstrList])
            self.rebuildDropdowns(adstrs)

        if self.window.lastPlotLinks is not None:
            newLinks = []
            for l in links:
                if not isinstance(l, tuple):
                    newLinks.append(l)
            self.rebuildPlotLinks(len(newLinks))
            for i,axis in enumerate(newLinks):
                for j in axis:
                    self.fcheckBoxes[i][j].setChecked(True)

    # returns list of list of strings (one list for each plot, (dstr, editNumber) for each trace)
    def getPlotInfo(self):
        dstrs = []
        if self.checkBoxMode:
            for cbAxis in self.checkBoxes:
                row = []
                for i,cb in enumerate(cbAxis):
                    if cb.isChecked():
                        # no easy way for current checkbox setup to work with edit number selection, so just use current
                        row.append((self.window.DATASTRINGS[i],self.window.currentEdit))
                dstrs.append(row)
        else:
            dditems = list(self.window.DATADICT.items())

            for ddRow in self.dropdowns:
                row = []
                for dd,en in ddRow: # for each dropdown in row
                    i = dd.currentIndex()-1
                    if i == -1: # dont include emptys
                        continue
                    for k,v in self.window.DATADICT.items(): # figure out which datastring is selected
                        if len(v[en]) > 0:
                            if i == 0:
                                row.append((k,en))
                                break
                            i -= 1
                dstrs.append(row)
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
            links.append((i,)) # append as tuple so can check for this later
        return links

    # rebuild all of the dropdowns whenever one changes because its too complicated to figure out
    # removing things correctly from qgridlayouts lol
    # empty dropdowns will have options based on current edit selection
    # filled dropdowns choices will be remembered and unavailable in subsequent dropdowns of same edit
    def rebuildDropdowns(self, overrideList = None):
        # first part of this function saves all the current strings selected
        dropList = []
        currentEN = max(self.ui.editCombo.currentIndex(),0)
        if isinstance(overrideList, list): # need this check because could be a single string from change callbacks
            for dstrs in overrideList:
                row = []
                for dstr in dstrs:
                    row.append(dstr)
                row.append(('',currentEN))
                dropList.append(row)
        else: # check what dropdowns are currently and rebuild (incase too many emptys or need to redo possible strs)
            for ddRow,lockedSet in zip(self.dropdowns,self.dropdownLocked):
                row = []
                for dd,en in ddRow:
                    idx = dd.currentIndex() - 1
                    if idx >= 0: # not empty selected then figure out what is original dstr
                        i = 0
                        for k,v in self.window.DATADICT.items():
                            if len(v[en]) == 0:
                                continue

                            #if (k,en) in lockedSet:
                            #    idx += 1

                            if i == idx:
                                row.append((k,en))
                                break
                            i += 1

                # always add empty to end of row
                row.append(('',currentEN))
                dropList.append(row)

            # ensure same number of rows as number of plots
            while len(dropList) < self.plotCount:
                dropList.append([('',currentEN)])
            if len(dropList) > self.plotCount:
                dropList = dropList[:(self.plotCount - len(dropList))]

        #print(dropList)

        self.checkBoxes = []
        self.dropdowns = []
        self.dropdownLocked = []
        PyQtUtils.clearLayout(self.ui.grid)

        for di,dropRow in enumerate(dropList):
            # add spacer in first row to take up right space
            if di == 0:
                spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
                self.ui.grid.addItem(spacer,0,100,1,1) # just so its always the last column
            # add plot label for row
            plotLabel = QtWidgets.QLabel()
            plotLabel.setText(f'Plot{di+1}')
            self.ui.grid.addWidget(plotLabel,di,0,1,1)
            lockedSet = set() # keeps track of options already chosen as tuple of (dstr,en)
            row = []
            for i,(dstr,en) in enumerate(dropRow):
                dd = QtWidgets.QComboBox()
                dd.addItem('')

                for k,v in self.window.DATADICT.items():
                    if len(v[en]) > 0:# and (k,en) not in lockedSet:
                        s = self.window.getLabel(k,en)
                        dd.addItem(s)
                        if k == dstr:
                            dd.setCurrentIndex(dd.count()-1)
                            lockedSet.add((k,en))

                dd.currentTextChanged.connect(self.rebuildDropdowns)
                self.ui.grid.addWidget(dd,di,i+1,1,1)
                row.append((dd,en))

            self.dropdowns.append(row)
            self.dropdownLocked.append(lockedSet)


    # callback for each checkbox on changed
    # which box, row and col are provided
    def checkPlotLinks(self, checkBox, r, c):
        #print(f'{r} {c}')
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
            pLabel.setText(f'Plot{i+1}')
            pLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            fgrid.addWidget(pLabel,0,i + 1,1,1)
        # add spacer
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        fgrid.addItem(spacer,0,self.plotCount + 1,1,1)

        # add checkBoxes
        for r in range(self.targRows):
            linkGroupLabel = QtWidgets.QLabel()
            linkGroupLabel.setText(f'Group{r+1}')
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

    def clearRows(self):
        if self.checkBoxMode:
            for row in self.checkBoxes:
                for cb in row:
                    cb.setChecked(False)
        else:
            override = []
            for row in self.dropdowns:
                override.append([])
            self.rebuildDropdowns(override)

    # returns bool matrix from checkbox matrix
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

    def addLabels(self):
        self.labels = []
        for i,dstr in enumerate(self.ABBRV_DSTRS):
            label = QtWidgets.QLabel()
            label.setText(dstr)
            label.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            self.ui.grid.addWidget(label,0,i + 1,1,1)

    def addPlot(self):
        self.plotCount += 1

        if self.checkBoxMode:
            plotLabel = QtWidgets.QLabel()
            plotLabel.setText(f'Plot{self.plotCount}')
            checkBoxes = []
            self.ui.grid.addWidget(plotLabel,self.plotCount + 1,0,1,1)
            for i,dstr in enumerate(self.ABBRV_DSTRS):
                checkBox = QtWidgets.QCheckBox()
                checkBoxes.append(checkBox)
                self.ui.grid.addWidget(checkBox,self.plotCount + 1,i + 1,1,1) # first +1 because axis labels
            self.checkBoxes.append(checkBoxes)
        else:
            self.rebuildDropdowns()

        self.rebuildPlotLinks()

    def removePlot(self):
        if self.plotCount <= 1:
            print('need at least one plot')
            return
        self.plotCount-=1

        if self.checkBoxMode:
            self.checkBoxes = self.checkBoxes[:-1]
            rowLen = len(self.ABBRV_DSTRS) + 1 # one extra because plot labels row
            for i in range(rowLen):
                child = self.ui.grid.takeAt(self.ui.grid.count() - 1) # take items off back
                if child.widget() is not None:
                    child.widget().deleteLater()
        else:
            self.rebuildDropdowns()

        self.ui.grid.invalidate() #otherwise gridframe doesnt shrink back down
        self.rebuildPlotLinks()        

