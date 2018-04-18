

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(250,200)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.clearButton = QtWidgets.QPushButton('Clear')

        layout.addWidget(self.clearButton)


class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)

        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self)
        
        #self.ui.clearButton.clicked.connect(self.clearCheckBoxes)

    # returns bool matrix from checkbox matrix
    def checksToBools(self, cbMatrix, skipEmpty = False):
        boolMatrix = []
        for cbAxis in cbMatrix:
            boolAxis = []
            nonEmpty = False
            for cb in cbAxis:
                b = cb.isChecked()
                nonEmpty = nonEmpty or b
                boolAxis.append(b)
            if nonEmpty or not skipEmpty:
                boolMatrix.append(boolAxis)
        return boolMatrix


