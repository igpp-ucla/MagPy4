
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import functools

class PlotTracerUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Plot Tracer')
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
        self.plotButton = QtWidgets.QPushButton('Plot')

        buttonLayout = QtWidgets.QGridLayout()
        buttonLayout.addWidget(self.clearButton, 0, 0, 1, 1)
        buttonLayout.addWidget(self.removePlotButton, 0, 1, 1, 1)
        buttonLayout.addWidget(self.addPlotButton, 0, 2, 1, 1)

        buttonLayout.addWidget(self.plotButton, 1, 0, 1, 3)

        self.switchButton = QtWidgets.QPushButton('Switch')
        buttonLayout.addWidget(self.switchButton, 0, 3, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        buttonLayout.addItem(spacer, 0, 4, 1, 1)

        self.layout.addLayout(buttonLayout)

        self.gridFrame = QtWidgets.QGroupBox('Plot Matrix')
        self.grid = QtWidgets.QGridLayout(self.gridFrame)
        self.layout.addWidget(self.gridFrame)

        self.fgridFrame = QtWidgets.QGroupBox('Y Axis Link Groups')
        self.fgrid = QtWidgets.QGridLayout(self.fgridFrame)
        self.layout.addWidget(self.fgridFrame)

        # make invisible stretch to take up rest of space
        self.layout.addStretch()

class PlotTracer(QtWidgets.QFrame, PlotTracerUI):
    def __init__(self, window, parent=None):
        super(PlotTracer, self).__init__(parent)

        self.window = window
        self.ui = PlotTracerUI()
        self.ui.setupUI(self)

        self.ui.clearButton.clicked.connect(self.clearRows)
        self.ui.addPlotButton.clicked.connect(self.addPlot)
        self.ui.removePlotButton.clicked.connect(self.removePlot)
        self.ui.plotButton.clicked.connect(self.plotData)
        self.ui.switchButton.clicked.connect(self.switchModes)

        self.checkBoxMode = True
        self.ui.switchButton.setText('Switch to Dropdowns' if self.checkBoxMode else 'Switch to CheckBoxes')
        self.fcheckBoxes = []

        self.shouldResizeWindow = False
        self.initTracer()


    def paintEvent(self, eve):
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

        self.initTracer()
        self.shouldResizeWindow = True
        

    def initTracer(self):
        self.plotCount = 0
        self.checkBoxes = []
        self.dropdowns = []
        self.clearLayout(self.ui.grid)

        if self.window.lastPlotStrings is None:
            print('no starting plot matrix!')
            return

        if self.checkBoxMode:
            self.addLabels()
            for i,axis in enumerate(self.window.lastPlotStrings):
                self.addPlot()
                for dstr in axis:
                    di = self.window.DATASTRINGS.index(dstr)
                    self.checkBoxes[i][di].setChecked(True)
        else:
            for i,axis in enumerate(self.window.lastPlotStrings):
                self.plotCount += 1
            self.rebuildDropdowns(self.window.lastPlotStrings)

        if self.window.lastPlotLinks is not None:
            links = []
            for l in self.window.lastPlotLinks:
                if not isinstance(l, tuple):
                    links.append(l)
            self.rebuildPlotLinks(len(links))
            for i,axis in enumerate(links):
                for j in axis:
                    self.fcheckBoxes[i][j].setChecked(True)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()

    # returns list of list of strings (one list for each plot, one string for each trace)
    def getPlotInfo(self):
        dstrs = []
        if self.checkBoxMode:
            for cbAxis in self.checkBoxes:
                row = []
                for i,cb in enumerate(cbAxis):
                    if cb.isChecked():
                        row.append(self.window.DATASTRINGS[i])
                dstrs.append(row)
        else:
            for ddAxis in self.dropdowns:
                row = []
                for i,dd in enumerate(ddAxis):
                    txt = dd.currentText()
                    if txt != '':
                        row.append(txt)
                dstrs.append(row)
        return dstrs

    def getLinkLists(self):
        links = []
        if len(self.fcheckBoxes) == 0:
            return links
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

    def plotData(self):
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

    # rebuild all of them whenever one changes because its too complicated to figure out
    # removing things correctly from qgridlayouts lol
    def rebuildDropdowns(self, overrideList = None):
        dropList = []
        if isinstance(overrideList, list): # incase its a single str from change callbacks
            for dstrs in overrideList:
                row = []
                for dstr in dstrs:
                    row.append(dstr)
                row.append('')
                dropList.append(row)
        else: # check what dropdowns are currently and rebuild (incase too many emptys or need to redo possible strs)
            for i,ddList in enumerate(self.dropdowns):
                row = []
                for dd in ddList:
                    txt = dd.currentText()
                    if txt != '':
                        row.append(txt)
                row.append('')
                dropList.append(row)

            while len(dropList) < self.plotCount:
                dropList.append([''])
            if len(dropList) > self.plotCount:
                dropList = dropList[:(self.plotCount - len(dropList))]

        self.checkBoxes = []
        self.dropdowns = []
        self.clearLayout(self.ui.grid)
        
        for di,drops in enumerate(dropList):
            strs = self.window.DATASTRINGS[:]
            for i,txt in enumerate(drops): # for each row in dropdowns, add marker next to ur current option
                for j,dstr in enumerate(strs):
                    if txt == dstr:
                        strs[j] = [txt,i]
            # add spacer in first row to take up right space
            if di == 0:
                spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
                self.ui.grid.addItem(spacer,0,100,1,1) # just so its always the last column
            # add plot label for row
            plotLabel = QtWidgets.QLabel()
            plotLabel.setText(f'Plot{di+1}')
            self.ui.grid.addWidget(plotLabel,di,0,1,1)
            row = []
            for i,txt in enumerate(drops):
                dd = QtWidgets.QComboBox()
                dd.addItem('')
                for s in strs:
                    if isinstance(s, list):
                        if s[1] == i: # if this is tagged as ur selected then add it
                            dd.addItem(s[0])
                            dd.setCurrentIndex(dd.count()-1)
                    else:
                        dd.addItem(s)
                dd.currentTextChanged.connect(self.rebuildDropdowns)
                self.ui.grid.addWidget(dd,di,i+1,1,1)
                row.append(dd)

            self.dropdowns.append(row)


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

        self.clearLayout(fgrid)

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
        for i,dstr in enumerate(self.window.DATASTRINGS):
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
            for i,dstr in enumerate(self.window.DATASTRINGS):
                checkBox = QtWidgets.QCheckBox()
                checkBoxes.append(checkBox)
                self.ui.grid.addWidget(checkBox,self.plotCount + 1,i + 1,1,1) # first +1 because axis labels
            self.checkBoxes.append(checkBoxes)
        else:
            self.rebuildDropdowns()

        self.rebuildPlotLinks()

    def removePlot(self):
        if self.plotCount == 0:
            print('no plots to delete')
            return
        self.plotCount-=1

        if self.checkBoxMode:
            self.checkBoxes = self.checkBoxes[:-1]
            rowLen = len(self.window.DATASTRINGS) + 1 # one extra because plot labels row
            for i in range(rowLen):
                child = self.ui.grid.takeAt(self.ui.grid.count() - 1) # take items off back
                if child.widget() is not None:
                    child.widget().deleteLater()
        else:
            self.rebuildDropdowns()

        self.ui.grid.invalidate() #otherwise gridframe doesnt shrink back down
        self.rebuildPlotLinks()        

