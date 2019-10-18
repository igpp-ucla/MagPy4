
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

    def getTimeStatusBar(self):
        layout = QtWidgets.QHBoxLayout()
        timeEdit = TimeEdit(QtGui.QFont())
        layout.addWidget(timeEdit.start)
        layout.addWidget(timeEdit.end)

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