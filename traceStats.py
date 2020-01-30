
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from MagPy4UI import PyQtUtils, TimeEdit
from scipy import stats

class TraceStatsUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Trace Stats')
        Frame.resize(300,200)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        cbLayout = QtWidgets.QHBoxLayout()
        self.onTopCheckBox = QtWidgets.QCheckBox()
        self.onTopCheckBox.setChecked(window.traceStatsOnTop)
        cbLabel = QtWidgets.QLabel('Stay on top')
        cbLayout.addWidget(self.onTopCheckBox)
        cbLayout.addWidget(cbLabel)
        cbLayout.addStretch()
        self.layout.addLayout(cbLayout)

        self.table = QtWidgets.QTableWidget()
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.layout.addWidget(self.table)

        # setup datetime edits
        timeFrame = QtWidgets.QGroupBox()
        timeLayout = QtWidgets.QVBoxLayout(timeFrame)
        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 10 if window.OS == 'windows' else 12))
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())
        timeLayout.addWidget(self.timeEdit.start)
        timeLayout.addWidget(self.timeEdit.end)

        self.layout.addWidget(timeFrame)

        # Setup button in trace stats window to crop main window to selected range
        selectedRngFrame = QtWidgets.QGroupBox()
        selectedRngLayout = QtWidgets.QHBoxLayout(selectedRngFrame)
        self.dispRangeBtn = QtGui.QPushButton('View Selection')
        self.dispRangeBtn.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,
            QSizePolicy.Minimum))
        selectedRngLayout.addWidget(self.dispRangeBtn)
        self.dtaBtn = QtGui.QPushButton('Data')
        self.dtaBtn.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,
            QSizePolicy.Minimum))
        selectedRngLayout.addWidget(self.dtaBtn)

        selectedRngLayout.addStretch(1)

        self.layout.addWidget(selectedRngFrame)

class TraceStats(QtWidgets.QFrame, TraceStatsUI):
    def __init__(self, window, parent=None):
        super(TraceStats, self).__init__(parent)

        self.window = window
        self.ui = TraceStatsUI()
        self.ui.setupUI(self, window)
        if self.ui.onTopCheckBox.isChecked():
            self.toggleWindowOnTop(True)
        self.ui.onTopCheckBox.clicked.connect(self.toggleWindowOnTop)
        self.ui.dispRangeBtn.clicked.connect(self.updtDispRange)
        self.ui.dtaBtn.clicked.connect(self.viewData)
        self.funcStrs = ['Min', 'Max', 'Mean', 'Median','Std Dev']
        self.funcs = [np.min, np.max, np.mean, np.median, np.std]

        self.clip = QtGui.QApplication.clipboard()
    
    def createRow(self, dta, prec):
        results = stats.describe(dta, axis=0, ddof=0)
        minVal, maxVal = results[1]
        mean = results[2]
        stdDev = np.sqrt(results[3])
        median = np.median(dta)
        toStr = lambda val : f'{val:.{prec}f}'
        row = list(map(toStr, [minVal, maxVal, mean, median, stdDev]))
        return row

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.window.traceStatsOnTop = self.ui.onTopCheckBox.isChecked()

    def toggleWindowOnTop(self, val):
        self.setParent(self.window if val else None)
        dialogFlag = QtCore.Qt.Dialog
        if self.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = self.windowFlags()
        flags = flags | dialogFlag if val else flags & ~dialogFlag
        self.setWindowFlags(flags)
        self.show()

    def viewData(self):
        self.window.showData()
        dataObj = self.window.tools['Data']
        if self.window.currentEdit > 0:
            dataObj.ui.viewEdtdDta.setChecked(True)
        startDt = self.ui.timeEdit.start.dateTime()
        dataObj.ui.dateTimeEdit.setDateTime(startDt)
        dataObj.moveByTime()

    def update(self):
        plotInfo = self.window.getSelectedPlotInfo()
       
        singleLine = (len(self.window.currSelect.regions) == 1 and self.window.currSelect.regions[0].isLine())
        colStrs = ['value'] if singleLine else self.funcStrs
        rowStrs = []

        grid = []
        prec = 6
        for dstrs,pens in plotInfo:
            group = []
            for i,(dstr,en) in enumerate(dstrs):
                indices = []
                for regNum in range(len(self.window.currSelect.regions)):
                    indices.append(self.window.calcDataIndicesFromLines(dstr, en, regNum))

                rowStrs.append([self.window.getLabel(dstr,en), pens[i].color().name()])
                row = []
                if singleLine:
                    row.append(f'{self.window.getData(dstr, en)[indices[0][0]]:.{prec}f}')
                else:
                    prunedData = []
                    for i0, i1 in indices:
                        # Merge data from all regions into single list
                        pruned = self.window.getPrunedData(dstr,en,i0,i1)
                        prunedData.extend(pruned)
                    # for func in self.funcs:
                    if len(prunedData) > 0:
                        row = self.createRow(prunedData, prec)
                    else:
                        row = ['---']*len(self.funcs)
                group.append(row)
            grid.append(group)

        self.ui.table.clearContents()
        self.ui.table.setRowCount(len(rowStrs))
        self.ui.table.setColumnCount(len(colStrs))
        for i,colStr in enumerate(colStrs):
            self.ui.table.setHorizontalHeaderItem(i,QtWidgets.QTableWidgetItem(colStr))
        rowIndex = -1
        altBackground = QtGui.QColor(230,230,230)
        for g,gridGroup in enumerate(grid):
            for gridRow in gridGroup:
                rowIndex += 1
                rowData = rowStrs[rowIndex]
                rowHeader = QtWidgets.QTableWidgetItem(rowData[0])
                rowHeader.setForeground(QtGui.QColor(rowData[1]))
                if g % 2 == 1:
                    rowHeader.setBackground(altBackground)
                self.ui.table.setVerticalHeaderItem(rowIndex,rowHeader)

                for c,s in enumerate(gridRow):
                    item = QtWidgets.QTableWidgetItem(s)
                    if g % 2 == 1:
                        item.setBackground(altBackground)
                    self.ui.table.setItem(rowIndex,c,item)

        self.ui.table.resizeColumnsToContents()
        self.ui.table.resizeRowsToContents()

        size = self.ui.layout.sizeHint()
        self.resize(size)

        # trying to do something with auto scrollbar if window tall enough but not working exactly
        #maxHeight = 300
        #if size.height() > maxHeight:
        #    self.ui.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        #    size = self.ui.layout.sizeHint()
        #    size = QtCore.QSize(size.width(),maxHeight)
        #else:
        #    self.ui.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def updtDispRange(self):
        # Uses trace stat's timeEdit values to update main's vals/sliders
        newStartVal = self.ui.timeEdit.start.dateTime()
        newEndVal = self.ui.timeEdit.end.dateTime()
        self.window.ui.timeEdit.start.setDateTime(newStartVal)
        self.window.ui.timeEdit.end.setDateTime(newEndVal)

    # this is for copy and paste functionality
    # todo: work on formatting here, oftentimes one tab is too short when strings are long
    # could just calculate purely with spaces
    def keyPressEvent(self, e):
        if (e.modifiers() & QtCore.Qt.ControlModifier):
            if e.key() == QtCore.Qt.Key_C: #copy
                selected = self.ui.table.selectedRanges()
                if selected:
                    s = '\t'+"\t".join([str(self.ui.table.horizontalHeaderItem(i).text()) for i in range(selected[0].leftColumn(), selected[0].rightColumn()+1)])
                    s = s + '\n'

                    for r in range(selected[0].topRow(), selected[0].bottomRow()+1):
                        s += self.ui.table.verticalHeaderItem(r).text() + '\t'
                        for c in range(selected[0].leftColumn(), selected[0].rightColumn()+1):
                            try:
                                s += str(self.ui.table.item(r,c).text()) + "\t"
                            except AttributeError:
                                s += "\t"
                        s = s[:-1] + "\n" #eliminate last '\t'
                    self.clip.setText(s)