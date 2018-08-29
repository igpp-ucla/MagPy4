
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from mth import Mth

class WaveAnalysisUI(object):
        def setupUI(self, Frame, window):
            Frame.setWindowTitle('Wave Analysis')
            Frame.resize(1000,700)  

            self.layout = QtWidgets.QVBoxLayout(Frame)

            self.axLayout = QtWidgets.QGridLayout()
            self.window = window
            defaultPlots = self.window.getDefaultPlotInfo()[0]
            axes = ['X','Y','Z','T']
            for i,ax in enumerate(axes):
                dd = QtGui.QComboBox()
                self.axLayout.addWidget(QtWidgets.QLabel(ax),0,i,1,1)
                for s in self.window.DATASTRINGS:
                    if ax.lower() in s.lower():
                        dd.addItem(s)
                dd.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                self.axLayout.addWidget(dd,1,i,1,1)

            spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.axLayout.addItem(spacer, 0, 100, 1, 1)

            self.layout.addLayout(self.axLayout)

            self.layout.addStretch()


class WaveAnalysis(QtWidgets.QFrame, WaveAnalysisUI):
    def __init__(self, window, parent=None):
        super(WaveAnalysis, self).__init__(parent)

        self.window = window
        self.ui = WaveAnalysisUI()
        self.ui.setupUI(self, window)