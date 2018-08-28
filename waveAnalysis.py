
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

class WaveAnalysisUI(object):
        def setupUI(self, Frame, window):
            Frame.setWindowTitle('Wave Analysis')
            Frame.resize(1000,700)  

class WaveAnalysis(QtWidgets.QFrame, WaveAnalysisUI):
    def __init__(self, window, parent=None):
        super(WaveAnalysis, self).__init__(parent)

        self.window = window
        self.ui = WaveAnalysisUI()
        self.ui.setupUI(self, window)