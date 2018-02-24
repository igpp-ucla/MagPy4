
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
from qrangeslider import QRangeSlider

#can add in translation stuff later

class UI_MagPy4(object):
    def setupUI(self, MagPy4):

        # gives default window options in top right
        MagPy4.setWindowFlags(QtCore.Qt.Window)
        MagPy4.resize(1500,1000)

        # a figure instance to plot on
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.canvas.mpl_connect('resize_event', MagPy4.resizeEvent)
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        #self.scroll = QtWidgets.QScrollArea()
        #self.scroll.setWidget(self.canvas)

        #self.scrollBar = NavigationToolbar(self.canvas, )

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar2QT(self.canvas, MagPy4)
        #self.toolbar.actions()[0].triggered.connect(tightness)
        #NavigationToolbar2QT.home = tightness

        # test buttons for functions
        self.button = QtWidgets.QPushButton('Plot')
        self.tightenButton = QtWidgets.QPushButton('Tighten')

        mainHoriz = QtWidgets.QHBoxLayout()
        mainHoriz.addWidget(self.canvas)
        
        self.comboBox = QtWidgets.QComboBox()
        self.comboBox.addItem("test1")
        self.comboBox.addItem("test2")
        self.comboBox.addItem("test3")
        self.comboBox.addItem("BX1")

        # horiz then vertical policy
        self.comboBox.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        mainHoriz.addWidget(self.comboBox)

        # main vertical layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)

        bottomHoriz = QtWidgets.QHBoxLayout()
        drawStyleLayout = QtWidgets.QHBoxLayout()
        self.drawStyleComboLabel = QtWidgets.QLabel()
        #self.drawStyleComboLabel.setAlignment
        self.drawStyleComboLabel.setText("Draw Style")
        self.drawStyleCombo = QtWidgets.QComboBox()
        self.drawStyleCombo.addItem('dots')
        self.drawStyleCombo.addItem('lines')
        drawStyleLayout.addWidget(self.drawStyleComboLabel)
        drawStyleLayout.addWidget(self.drawStyleCombo)
        self.drawStyleCombo.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.drawStyleComboLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #layout.addLayout(drawStyleLayout)
        bottomHoriz.addLayout(drawStyleLayout)

        bottomHoriz.addWidget(self.button)
        bottomHoriz.addWidget(self.tightenButton)

        layout.addLayout(bottomHoriz)

        layout.addLayout(mainHoriz)
        #layout.addWidget(self.canvas)
        #layout.addWidget(self.scroll)

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

        #self.timeSlider = QRangeSlider()
        #self.timeSlider.setOrientation(QtCore.Qt.Horizontal)
        #layout.addWidget(self.timeSlider)


        #layout.addWidget()



        MagPy4.setLayout(layout)
