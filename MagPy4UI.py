
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

        self.centralWidget = QtWidgets.QWidget(MagPy4)
        MagPy4.setCentralWidget(self.centralWidget)

        self.toolBar = QtWidgets.QToolBar(MagPy4)
        MagPy4.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MagPy4)
        self.actionOpen.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpen.setText('Open')
        self.actionOpen.setShortcut('O')

        self.actionTest = QtWidgets.QAction(MagPy4)
        self.actionTest.setText('Test')

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionTest)

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
        #self.toolbar = NavigationToolbar2QT(self.canvas, MagPy4)
        #self.toolbar.actions()[0].triggered.connect(tightness)
        #NavigationToolbar2QT.home = tightness

        mainHoriz = QtWidgets.QHBoxLayout()
        mainHoriz.addWidget(self.canvas)
        
        # main vertical layout
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

        #datas = ['BX1','BX2','BX3','BX4','BY1','BY2','BY3','BY4','BZ1','BZ2','BZ3','BZ4','BT1','BT2','BT3','BT4','curl','velocity','pressure','density']
        #axisCount = 4
        #for a in range(axisCount+1):
        #    if a == 0: # make labels
        #        for i,dstr in enumerate(datas):
        #            label = QtWidgets.QLabel()
        #            label.setText(dstr)
        #            label.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        #            self.grid.addWidget(label,a,i+1,1,1)
        #    else:
        #        axLabel = QtWidgets.QLabel()
        #        axLabel.setText(f'Axis{a}')
        #        self.grid.addWidget(axLabel,a,0,1,1)
        #        for i,dstr in enumerate(datas):
        #            checkBox = QtWidgets.QCheckBox()
        #            #checkBox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred))
        #            self.grid.addWidget(checkBox,a,i+1,1,1)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.gridFrame = QtWidgets.QGroupBox('Axes Matrix')
        self.grid = QtWidgets.QGridLayout(self.gridFrame)

        buttonLayout = QtWidgets.QHBoxLayout()
        layout.addLayout(buttonLayout)

        drawStyleLayout = QtWidgets.QHBoxLayout()
        drawStyleComboLabel = QtWidgets.QLabel()
        drawStyleComboLabel.setText("Draw Style")
        self.drawStyleCombo = QtWidgets.QComboBox() # add this to AxisTracer instead
        self.drawStyleCombo.addItem('dots')
        self.drawStyleCombo.addItem('lines')
        drawStyleLayout.addWidget(drawStyleComboLabel)
        drawStyleLayout.addWidget(self.drawStyleCombo)
        self.drawStyleCombo.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        drawStyleComboLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        buttonLayout.addLayout(drawStyleLayout)

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
