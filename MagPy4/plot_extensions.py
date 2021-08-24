import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
class GridGraphicsLayout(pg.GraphicsLayout):
    def __init__(self, window=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self.lastWidth = 0
        self.lastHeight = 0

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        """
        Add an item to the layout and place it in the next available cell (or in the cell specified).
        The item must be an instance of a QGraphicsWidget subclass.
        """
        if row is None:
            row = self.currentRow
        if col is None:
            col = self.currentCol
            
        self.items[item] = []
        for i in range(rowspan):
            for j in range(colspan):
                row2 = row + i
                col2 = col + j
                if row2 not in self.rows:
                    self.rows[row2] = {}
                self.rows[row2][col2] = item
                self.items[item].append((row2, col2))

        borderRect = QtWidgets.QGraphicsRectItem()

        borderRect.setParentItem(self)
        borderRect.setZValue(1e3)
        borderRect.setPen(pg.mkPen(self.border))

        self.itemBorders[item] = borderRect

        if hasattr(item, 'geometryChanged'):
            item.geometryChanged.connect(self._updateItemBorder)

        self.layout.addItem(item, row, col, rowspan, colspan)
                               # Allows some PyQtGraph features to also work without Qt event loop.
        
        self.nextColumn()
