
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

        layout.addLayout(mainHoriz)
        #layout.addWidget(self.canvas)
        #layout.addWidget(self.scroll)

        self.startSlider = QtWidgets.QSlider()
        self.startSlider.setOrientation(QtCore.Qt.Horizontal)
        layout.addWidget(self.startSlider)
        self.endSlider = QtWidgets.QSlider()
        self.endSlider.setOrientation(QtCore.Qt.Horizontal)
        layout.addWidget(self.endSlider)

        #self.timeSlider = QRangeSlider()
        #self.timeSlider.setOrientation(QtCore.Qt.Horizontal)
        #layout.addWidget(self.timeSlider)


        #layout.addWidget()

        horiz = QtWidgets.QHBoxLayout()
        horiz.addWidget(self.button)
        horiz.addWidget(self.tightenButton)
        layout.addLayout(horiz)

        MagPy4.setLayout(layout)
