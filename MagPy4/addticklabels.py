from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import functools
from scipy import interpolate

class ColumnLayout(QtWidgets.QGridLayout):
    def __init__(self, numCols=1, *args, **kwargs):
        self.numCols = numCols
        self.numItems = 0
        self.row = 0
        self.col = 0
        QtWidgets.QGridLayout.__init__(self, *args, **kwargs)

    def addWidget(self, widget):
        if self.col >= self.numCols:
            self.col = 0
            self.row += 1

        QtWidgets.QGridLayout.addWidget(self, widget, self.row, self.col, 1, 1)
        self.col += 1
        self.numItems += 1

class AddTickLabelsUI(object):
    def setupUI(self, Frame, dstrList):
        Frame.setWindowTitle('Additional Tick Labels')
        Frame.resize(100, 100)
        wrapLt = QtWidgets.QVBoxLayout(Frame)

        # Set up options frame
        settingsLt = QtWidgets.QGridLayout()
        self.locBox = QtWidgets.QComboBox()
        self.locBox.addItems(['Top', 'Bottom'])
        lbl = QtWidgets.QLabel('Location: ')
        settingsLt.addWidget(lbl, 0, 0, 1, 1)
        settingsLt.addWidget(self.locBox, 0, 1, 1, 1)
        self.locBox.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        wrapLt.addLayout(settingsLt)

        # Create a scroll frame to wrap column layout in
        scrollFrame = QtWidgets.QScrollArea()
        scrollFrame.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        scrollFrame.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        wrapLt.addWidget(scrollFrame)

        # Create and wrap column layout in scroll frame
        wrapFrame = QtWidgets.QFrame()
        layout = ColumnLayout(4, wrapFrame)

        scrollFrame.setWidget(wrapFrame)
        scrollFrame.setWidgetResizable(True)
        wrapFrame.setStyleSheet('.QFrame { background-color: #fafafa; }')

        # Initialize checkbox elements for all dstrs not plotted or with
        # tick labels set
        self.chkboxes = []
        for dstr in dstrList:
            chkbx = QtWidgets.QCheckBox(dstr)
            layout.addWidget(chkbx)
            self.chkboxes.append(chkbx)

        # Adjust wrap frame minimum width
        minWidth = wrapFrame.minimumWidth()
        wrapFrame.setMinimumWidth(minWidth + 20)

class AddTickLabels(QtWidgets.QFrame, AddTickLabelsUI):
    def __init__(self, pltGrd, window, parent=None):
        super(AddTickLabels, self).__init__(parent)
        self.ui = AddTickLabelsUI()
        self.window = window
        self.pltGrd = pltGrd
        if not isinstance(self.window, QtWidgets.QMainWindow):
            self.valid = False
            return
        else:
            self.valid = True

        # Set up UI
        dstrs = list(self.window.DATADICT.keys())
        self.ui.setupUI(self, dstrs)

        # Set default as last set location
        self.ui.locBox.setCurrentIndex(1)

        # Connect every checkbox to function
        for chkbx in self.ui.chkboxes:
            chkbx.clicked.connect(functools.partial(self.addLabelSet, chkbx))
        self.ui.locBox.currentTextChanged.connect(self.loc_changed)
        self.loc_changed()

    def get_loc(self):
        ''' Return the location where axes are being added '''
        return self.ui.locBox.currentText().lower()

    def addLabelSet(self, chkbx):
        # Get axis name and location
        dstr = chkbx.text()
        loc = self.get_loc()
        checked = chkbx.isChecked()

        # Remove or create axis item
        if checked:
            # Create interpolator function
            en = self.window.currentEdit
            x = self.window.getTimes(dstr, en)[0]
            y = self.window.getData(dstr, en)
            interp = interpolate.interp1d(x, y, bounds_error=False, fill_value=self.window.errorFlag)

            # Add axis item to grid
            self.pltGrd.grid.add_axis(dstr, interp, loc)
        else:
            # Remove axis item
            self.pltGrd.grid.remove_axis(dstr, loc)
        
    def set_checked_items(self, prev_dstrs):
        ''' Set checked items to reflect previously added axis items '''
        for box in self.ui.chkboxes:
            check = (box.text() in prev_dstrs)
            box.blockSignals(True)
            box.setChecked(check)
            box.blockSignals(False)

    def loc_changed(self):
        # Get current axis location
        loc = self.get_loc()

        # Get list of all dstrs that currently have extra tick labels set
        prev_dstrs = self.pltGrd.list_axis_grids()[loc]

        # Check list of items
        self.set_checked_items(prev_dstrs)
