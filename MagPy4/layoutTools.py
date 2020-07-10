
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .pyqtgraphExtensions import GridGraphicsLayout
import pyqtgraph as pg
import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from .MagPy4UI import TimeEdit, NumLabel

class BaseLayout(object):
    def __init__(self):
        self.app = QtCore.QCoreApplication.instance()

    def processEvents(self):
        self.app.processEvents()

    def getSizePolicy(self, horz, vert):
        if horz == 'Min':
            horz = QSizePolicy.Minimum
        elif horz == 'Max':
            horz = QSizePolicy.Maximum
        elif horz == 'Exp':
            horz = QSizePolicy.Expanding
        elif horz == 'MinExp':
            horz = QSizePolicy.MinimumExpanding
        else:
            horz = QSizePolicy.Preferred

        if vert == 'Min':
            vert = QSizePolicy.Minimum
        elif vert == 'Max':
            vert = QSizePolicy.Maximum
        elif vert == 'Exp':
            vert = QSizePolicy.Expanding
        elif vert == 'MinExp':
            vert = QSizePolicy.MinimumExpanding
        else:
            vert = QSizePolicy.Preferred

        return QSizePolicy(horz, vert)

    def getTimeStatusBar(self, optWidgets=[]):
        layout = QtWidgets.QHBoxLayout()
        timeEdit = TimeEdit(QtGui.QFont())
        layout.addWidget(timeEdit.start)
        layout.addWidget(timeEdit.end)

        for widget in optWidgets:
            layout.addWidget(widget)

        layout.addItem(self.getSpacer(5))

        statusBar = QtWidgets.QStatusBar()
        layout.addWidget(statusBar)
        timeEdit.start.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        timeEdit.end.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        statusBar.setSizePolicy(self.getSizePolicy('Min', 'Max'))

        return layout, timeEdit, statusBar

    def addPair(self, layout, name, elem, row, col, rowspan, colspan, tooltip=None):
        # Create a label for given widget and place both into layout
        lbl = QtWidgets.QLabel(name)
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        if name != '':
            layout.addWidget(lbl, row, col, 1, 1)
        layout.addWidget(elem, row, col+1, rowspan, colspan)

        # Set any tooltips if given
        if tooltip is not None:
            lbl.setToolTip(tooltip)

        return lbl

    def getMaxLabelWidth(label, gwin):
        # Gives the maximum number of characters that should, on average, fit
        # within the width of the window the label is in
        lblFont = label.font()
        fntMet = QtGui.QFontMetrics(lblFont)
        avgCharWidth = fntMet.averageCharWidth()
        winWidth = gwin.width()
        return winWidth / avgCharWidth
    
    def getSpacer(self, width):
        spacer = QtWidgets.QSpacerItem(width, 1)
        return spacer

    def getGraphicSpacer(self, horz='Exp', vert='Exp'):
        spacer = pg.LabelItem('')
        spacer.setSizePolicy(self.getSizePolicy(horz, vert))
        return spacer

    def getGraphicsGrid(self, window=None):
        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        return self.glw

# Subclass of Qt's Table Widget w/ a simplified interface for adding
# rows to a table
class TableWidget(QtWidgets.QTableWidget):
    def __init__(self, numCols, parent=None):
        super().__init__(parent)
        for i in range(0, numCols):
            self.insertColumn(0)

        # Selecting a cell selects the entire row
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        # Columns should stretch to fill available space + don't highlight names
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        header.setHighlightSections(False)

        # Add a copy shortcut
        copyShrtct = QtWidgets.QShortcut('Ctrl+c', self)
        copyShrtct.activated.connect(self.copyData)
    
    def setTableData(self, table):
        table = np.array(table)
        rows, cols = table.shape
        for i in range(0, rows):
            self.addRowItem(table[row])

    def setHeader(self, colNames):
        if len(colNames) != self.columnCount():
            return

        self.setHorizontalHeaderLabels(colNames)
    
    def addRowItem(self, item, row=None):
        if len(item) != self.columnCount():
            return
        
        row = self.rowCount() if row is None else row
        self.insertRow(row)
        self.setRowItem(row, item)

    def setRowItem(self, row, item):
        if len(item) != self.columnCount() or row < 0 or row >= self.count():
            return

        for col in range(0, self.columnCount()):
            itemText = str(item[col])
            tableItem = QtWidgets.QTableWidgetItem()
            tableItem.setText(itemText)
            self.setItem(row, col, tableItem)

    def count(self):
        return self.rowCount()
    
    def setCurrentRow(self, row):
        self.setCurrentCell(row, 0)
    
    def getSelectedRows(self):
        ranges = self.selectedRanges()

        # Get lower/upper row bounds for each selected range
        # and add each row in between them (inclusive) to the row list
        rows = []
        for rangeObj in ranges:
            botmRow = rangeObj.bottomRow()
            topRow = rangeObj.topRow()
            botmRow, topRow = min(botmRow, topRow), max(botmRow, topRow)
            rows.extend([i for i in range(botmRow, topRow+1)])

        return list(set(rows)) # Remove duplicates

    def getRowItem(self, row):
        rowItems = []
        for col in range(0, self.columnCount()):
            rowItems.append(self.item(row, col).text())
        return rowItems

    def copyData(self):
        rows = self.getSelectedRows()
        rows.sort()

        data = []
        for row in rows:
            rowItems = self.getRowItem(row)
            if len(rowItems) > 0:
                data.append(', '.join(rowItems))

        if len(data) > 0:
            data = '\n'.join(data)
            QtGui.QApplication.clipboard().setText(data)
