
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from pyqtgraphExtensions import GridGraphicsLayout
import pyqtgraph as pg
import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from MagPy4UI import TimeEdit, NumLabel

class BaseLayout(object):
    def getSizePolicy(self, horz, vert):
        if horz == 'Min':
            horz = QSizePolicy.Minimum
        else:
            horz = QSizePolicy.Maximum

        if vert == 'Min':
            vert = QSizePolicy.Minimum
        else:
            vert = QSizePolicy.Maximum

        return QSizePolicy(horz, vert)

    def getTimeStatusBar(self):
        layout = QtWidgets.QHBoxLayout()
        timeEdit = TimeEdit(QtGui.QFont())
        layout.addWidget(timeEdit.start)
        layout.addWidget(timeEdit.end)
        statusBar = QtWidgets.QStatusBar()
        layout.addWidget(statusBar)

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

    def getGraphicsGrid(self, window):
        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        return self.glw