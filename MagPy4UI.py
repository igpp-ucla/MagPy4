
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg

#from qrangeslider import QRangeSlider
#can add in translation stuff later

class UI_MagPy4(object):
    def setupUI(self, MagPy4):

        # gives default window options in top right
        MagPy4.setWindowFlags(QtCore.Qt.Window)
        MagPy4.resize(1500,1000)

        self.centralWidget = QtWidgets.QWidget(MagPy4)
        MagPy4.setCentralWidget(self.centralWidget)

        self.toolBar = QtWidgets.QToolBar(MagPy4)
        MagPy4.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MagPy4)
        self.actionOpen.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpen.setText('Open')
        self.actionOpen.setShortcut('O')

        self.actionPlot = QtWidgets.QAction(MagPy4)
        self.actionPlot.setText('Plot Menu')

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionPlot)

        self.glw = pg.GraphicsLayoutWidget()#border=(100,100,100))

        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        mainHoriz = QtWidgets.QHBoxLayout()
        mainHoriz.addWidget(self.glw)
        
        layout = QtWidgets.QVBoxLayout(self.centralWidget)

        layout.addLayout(mainHoriz)

        # SLIDER setup
        sliderFont = QtGui.QFont("Times", 14)#, QtGui.QFont.Bold) 
        sliderLayout = QtWidgets.QGridLayout() # r, c, w, h
        self.startSliderLabel = QtWidgets.QLabel()
        self.startSliderLabel.setText("StartTime")
        self.startSliderLabel.setFont(sliderFont)
        self.startSlider = QtWidgets.QSlider()
        self.startSlider.setOrientation(QtCore.Qt.Horizontal)
        self.endSliderLabel = QtWidgets.QLabel()
        self.endSliderLabel.setText("EndTime")
        self.endSliderLabel.setFont(sliderFont)
        self.endSlider = QtWidgets.QSlider()
        self.endSlider.setOrientation(QtCore.Qt.Horizontal)
        sliderLayout.addWidget(self.startSliderLabel, 0, 0, 1, 1)
        sliderLayout.addWidget(self.startSlider, 0, 1, 1, 1)
        sliderLayout.addWidget(self.endSliderLabel, 1, 0, 1, 1)
        sliderLayout.addWidget(self.endSlider, 1, 1, 1, 1)

        layout.addLayout(sliderLayout)

 
class UI_AxisTracer(object):
    def setupUI(self, Frame):
        Frame.resize(500,500)

        checkBoxStyle = """
            QCheckBox{spacing: 0px;}
            QCheckBox::indicator{width:32px;height:32px}
            QCheckBox::indicator:unchecked {            image: url(images/checkbox_unchecked.png);}
            QCheckBox::indicator:unchecked:hover {      image: url(images/checkbox_unchecked_hover.png);}
            QCheckBox::indicator:unchecked:pressed {    image: url(images/checkbox_unchecked_pressed.png);}
            QCheckBox::indicator:checked {              image: url(images/checkbox_checked.png);}
            QCheckBox::indicator:checked:hover {        image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:checked:pressed {      image: url(images/checkbox_checked_pressed.png);}
            QCheckBox::indicator:indeterminate:hover {  image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:indeterminate:pressed {image: url(images/checkbox_checked_pressed.png);}
        """

        Frame.setStyleSheet(checkBoxStyle)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.gridFrame = QtWidgets.QGroupBox('Axes Matrix')
        self.grid = QtWidgets.QGridLayout(self.gridFrame)

        buttonLayout = QtWidgets.QHBoxLayout()
        layout.addLayout(buttonLayout)

        #drawStyleLayout = QtWidgets.QHBoxLayout()
        #drawStyleComboLabel = QtWidgets.QLabel()
        #drawStyleComboLabel.setText("Draw Style")
        #self.drawStyleCombo = QtWidgets.QComboBox() # add this to AxisTracer instead
        #self.drawStyleCombo.addItem('dots')
        #self.drawStyleCombo.addItem('lines')
        #drawStyleLayout.addWidget(drawStyleComboLabel)
        #drawStyleLayout.addWidget(self.drawStyleCombo)
        #self.drawStyleCombo.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #drawStyleComboLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #buttonLayout.addLayout(drawStyleLayout)

        fixedLabel = QtWidgets.QLabel()
        fixedLabel.setText("Fixed Y Axis")
        fixedLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.fixedAxisCheckBox = QtWidgets.QCheckBox()
        self.fixedAxisCheckBox.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        buttonLayout.addWidget(fixedLabel)
        buttonLayout.addWidget(self.fixedAxisCheckBox)

        self.clearButton = QtWidgets.QPushButton('Clear')
        buttonLayout.addWidget(self.clearButton)
        self.removeAxisButton = QtWidgets.QPushButton('Remove Axis')
        buttonLayout.addWidget(self.removeAxisButton)
        self.addAxisButton = QtWidgets.QPushButton('Add Axis')
        buttonLayout.addWidget(self.addAxisButton)
        self.plotButton = QtWidgets.QPushButton('Plot')
        layout.addWidget(self.plotButton)

        layout.addWidget(self.gridFrame)

        # take up the rest of the space, otherwise top label row in grid will. not sure how to do this otherwise
        spacelabel = QtWidgets.QLabel()
        spacelabel.setText(' ')
        spacelabel.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout.addWidget(spacelabel)
