
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from .MagPy4UI import TimeEdit

class LabeledProgress(QtWidgets.QProgressBar):
    ''' Displays a progress bar with text overlaying it '''
    def __init__(self, txt=''):
        self.txt = txt
        super().__init__()

    def paintEvent(self, ev):
        # Create a painter and draw text in center of rect
        QtWidgets.QProgressBar.paintEvent(self, ev)
        p = QtGui.QPainter(self)
        pen = pg.mkPen(255, 255, 255)
        p.setPen(pen)
        p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.txt)
        p.end()

    def setText(self, txt):
        self.txt = txt

class BoxLayout():
    ''' Superclass for HBoxLayout and VBoxLayout that adds
        some additional useful operations for accessing 
        widgets and removing items
    '''
    def getItems(self):
        ''' Returns all widgets in layout '''
        n = self.count()
        items = [self.itemAt(i) for i in range(n)]
        widgets = [item.widget() for item in items]
        return widgets

    def clear(self):
        ''' Removes and deletes all elements from layout '''
        items = self.getItems()
        for item in items:
            self.removeWidget(item)
            item.deleteLater()

    def pop(self, index=None):
        ''' Remove and delete last (or specified) element from layout '''
        if index is None:
            n = self.count() - 1
        else:
            n = index
        
        # Remove item and delete
        item = self.itemAt(n).widget()
        self.removeWidget(item)
        item.deleteLater()

class HBoxLayout(QtWidgets.QHBoxLayout, BoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class VBoxLayout(QtWidgets.QVBoxLayout, BoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        timeEdit = TimeEdit()
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
        from .plotuibase import GraphicsView
        self.gview = GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = pg.GraphicsLayout()
        self.gview.setCentralItem(self.glw)
        return self.glw

class TableItem(QtWidgets.QTableWidgetItem):
    def value(self):
        return self.data(QtCore.Qt.UserRole)
    
    def setValue(self, value):
        self.setData(QtCore.Qt.UserRole, value)

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
        ''' Sets table as data for table '''
        table = np.array(table)
        rows, cols = table.shape
        for i in range(0, rows):
            self.addRowItem(table[row])

    def setHeader(self, colNames):
        ''' Sets labels for columns '''
        if len(colNames) != self.columnCount():
            return

        self.setHorizontalHeaderLabels(colNames)
    
    def addRowItem(self, item, data=None, row=None):
        ''' Adds a row and inserts item into row '''
        if len(item) != self.columnCount():
            return
        
        # Get item row and add a row to table
        row = self.rowCount() if row is None else row
        self.insertRow(row)

        # Set row item and data if given
        self.setRowItem(row, item)
        if data is not None:
            self.setRowData(row, data)

    def setRowItem(self, row, item):
        ''' Sets item text values for each item in row '''
        if len(item) != self.columnCount() or row < 0 or row >= self.count():
            return

        for col in range(0, self.columnCount()):
            itemText = str(item[col])
            tableItem = TableItem()
            tableItem.setText(itemText)
            self.setItem(row, col, tableItem)
    
    def setRowData(self, row, data):
        ''' Sets the internal item data for the given row '''
        if row < 0 or row >= self.count():
            return
        
        # Get each item in row and set its corresponding internal data item
        for col in range(0, self.columnCount()):
            item_data = data[col]
            item = self.item(row, col)
            item.setValue(item_data)

    def count(self):
        ''' Number of rows in table '''
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
        ''' Returns a list of text values for each item in given row '''
        rowItems = []
        for col in range(0, self.columnCount()):
            rowItems.append(self.item(row, col).text())
        return rowItems

    def getRowData(self, row):
        ''' Returns a list of internal data objects for each item in given row '''
        rowItems = []
        for col in range(0, self.columnCount()):
            item = self.item(row, col)
            data = item.value()
            rowItems.append(data)
        
        return rowItems

    def copyData(self):
        ''' Gets selected rows and adds to clipboard '''
        rows = self.getSelectedRows()
        rows.sort()

        data = []
        for row in rows:
            rowItems = self.getRowItem(row)
            if len(rowItems) > 0:
                data.append(', '.join(rowItems))

        if len(data) > 0:
            data = '\n'.join(data)
            QtWidgets.QApplication.clipboard().setText(data)

    def removeRow(self, row):
        ''' Removes row without sending out signals
        '''
        # Remove row from layout
        self.blockSignals(True)
        super().removeRow(row)
        self.blockSignals(False)

    def removeRows(self, rows):
        ''' Removes each row from a given list of row numbers '''
        # Remove duplicates and sort row numbers in reverse
        # order so there isn't a need to adjust indices as
        # rows are removed
        rows = sorted(list(set(rows)), reverse=True)
        for row in rows:
            self.removeRow(row)

        # Update row to next row item if it exists
        if (len(rows) > 0) and rows[-1] < self.rowCount():
            self.selectRow(rows[-1])

    def removeSelected(self):
        ''' Remove selected rows from table '''
        selected = self.getSelectedRows()
        self.removeRows(selected)

class SplitHandle(QtWidgets.QSplitterHandle):
    ''' SplitterHandle with modified appearance '''
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.setCursor(QtCore.Qt.SplitVCursor)

    def paintEvent(self, ev):
        ''' Draws a gray line across splitter region '''
        if not self.isVisible():
            return

        # Set up painter and pen
        painter = QtGui.QPainter(self)
        pen = pg.mkPen(200, 200, 200)
        pen.setWidth(2)
        painter.setPen(pen)

        # Get rect width
        rect = self.rect()
        y = rect.center().y()
        start = rect.left()
        stop = rect.right()

        # Draw line
        painter.drawLine(start, y, stop, y)
        painter.end()
    
class SplitterWidget(QtWidgets.QSplitter):
    ''' Splitter with modified SplitterHandle appearance '''
    def createHandle(self):
        orientation = self.orientation()
        return SplitHandle(orientation, self)