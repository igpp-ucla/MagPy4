
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg

# based off class here
#https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/graphicsItems/GraphicsLayout.py
# ref for qt layout component
#http://doc.qt.io/qt-5/qgraphicslinearlayout.html
__all__ = ['GraphicsLayout']
class LinearGraphicsLayout(pg.GraphicsWidget):
    """
    Used for laying out GraphicsWidgets in a linear fashion
    """

    def __init__(self, orientation=QtCore.Qt.Vertical, parent=None):
        pg.GraphicsWidget.__init__(self, parent)
        self.layout = QtGui.QGraphicsLinearLayout(orientation)
        self.setLayout(self.layout)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        self.items = []
        
    def addLayout(self, **kargs):
        """
        Create an empty GraphicsLayout and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`GraphicsLayout.__init__ <pyqtgraph.GraphicsLayout.__init__>`
        Returns the created item.
        """
        layout = LinearGraphicsLayout(QtCore.Qt.Horizontal, **kargs)
        self.addItem(layout)
        return layout
        
    def addItem(self, item):
        self.items.append(item)
        self.layout.addItem(item)

    def itemIndex(self, item):
        for i in range(self.layout.count()):
            if self.layout.itemAt(i).graphicsItem() is item:
                return i
        raise Exception("Could not determine index of item " + str(item))

    def removeItem(self, item):
        """Remove *item* from the layout."""
        ind = self.itemIndex(item)
        self.layout.removeAt(ind)
        self.scene().removeItem(item)
        self.items = [x for x in self.items if x != item]
        self.update()

    def clear(self):
        for i in self.items:
            self.removeItem(i)

    def setContentsMargins(self, *args):
        # Wrap calls to layout. This should happen automatically, but there
        # seems to be a Qt bug:
        # http://stackoverflow.com/questions/27092164/margins-in-pyqtgraphs-graphicslayout
        self.layout.setContentsMargins(*args)

    def setSpacing(self, *args):
        self.layout.setSpacing(*args)
    