
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from MagPy4UI import PyQtUtils, TimeEdit

class TraceStatsUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Trace Stats')
        Frame.resize(300,200)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        cbLayout = QtWidgets.QHBoxLayout()
        self.onTopCheckBox = QtWidgets.QCheckBox()
        cbLabel = QtWidgets.QLabel('Stay on top')
        cbLayout.addWidget(self.onTopCheckBox)
        cbLayout.addWidget(cbLabel)
        cbLayout.addStretch()
        self.layout.addLayout(cbLayout)

        self.gridLayout = QtWidgets.QGridLayout()
        self.layout.addLayout(self.gridLayout)

        timeFrame = QtWidgets.QGroupBox()
        timeLayout = QtWidgets.QVBoxLayout(timeFrame)

        # setup datetime edits
        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 10 if window.OS == 'windows' else 14))
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        timeLayout.addWidget(self.timeEdit.start)
        timeLayout.addWidget(self.timeEdit.end)
        self.layout.addWidget(timeFrame)


class TraceStats(QtWidgets.QFrame, TraceStatsUI):
    def __init__(self, window, plotIndex, parent=None):
        super(TraceStats, self).__init__(parent)

        self.window = window
        self.plotIndex = plotIndex
        self.ui = TraceStatsUI()
        self.ui.setupUI(self, window)
        self.ui.onTopCheckBox.clicked.connect(self.toggleWindowHint)

        self.penColors = [p.color().name() for p in self.window.plotTracePens[self.plotIndex]]
        self.funcStrs = ['','start','min', 'max', 'mean', 'median','std dev']
        self.funcs = [np.min, np.max, np.mean, np.median, np.std]

    def closeEvent(self, event):
        self.window.endGeneralSelect()

    def toggleWindowHint(self, val):
        flags = self.windowFlags()
        toggleFlag = QtCore.Qt.WindowStaysOnTopHint
        flags = flags | toggleFlag if val else flags & ~toggleFlag
        self.setWindowFlags(flags)
        self.show()

    def onChange(self):
        # clear layout
        PyQtUtils.clearLayout(self.ui.gridLayout)

        i0,i1 = self.window.getTicksFromLines()
        print(f'{i0} {i1}')

        #if self.window.generalSelectStep == 1:
        #    return

        dstrs = self.window.lastPlotStrings[self.plotIndex]
       
        grid = [[self.funcStrs, '#000000']]
        prec = 6
        for i,dstr in enumerate(dstrs):
            pruned = self.window.getPrunedData(dstr,i0,i1)
            row = [dstr]
            row.append(f'{self.window.getData(dstr)[i0]:.{prec}f}')
            if self.window.generalSelectStep > 1:
                for func in self.funcs:
                    if len(pruned) > 0:
                        row.append(f'{func(pruned):.{prec}f}')
                    else:
                        row.append('---')
            grid.append([row, self.penColors[i]])

        for r,gridRow in enumerate(grid):
            color = gridRow[1]
            for c,s in enumerate(gridRow[0]):
                if not s:
                    continue
                lab = QtWidgets.QLabel()
                lab.setText(s)
                lab.setStyleSheet(f'color:{color}')
                lab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
                self.ui.gridLayout.addWidget(lab, r, c, 1, 1)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ui.gridLayout.addItem(spacer, 100, 100, 1, 1)