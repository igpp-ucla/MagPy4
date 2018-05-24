
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from mth import Mth

class EditUI(object):

    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')

        w = 600 if window.OS == 'windows' else 800
        Frame.resize(w,350)

        mainLayout = QtWidgets.QVBoxLayout(Frame)
        upperLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        rightLayout = QtWidgets.QVBoxLayout()
        upperLayout.addLayout(leftLayout,1)
        upperLayout.addLayout(rightLayout,1)
        mainLayout.addLayout(upperLayout)

        # this part gets built dynamically
        vectorFrame = QtWidgets.QGroupBox('Data Vectors')
        self.vectorLayout = QtWidgets.QVBoxLayout(vectorFrame)
        leftLayout.addWidget(vectorFrame)

        # axis rotation frame
        builderFrame = QtWidgets.QGroupBox('Matrix Builders')

        builderLayout = QtWidgets.QVBoxLayout(builderFrame)

        self.manRotButton = QtGui.QPushButton('Custom Rotation')
        self.manRotButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        builderLayout.addWidget(self.manRotButton)

        self.minVarButton = QtGui.QPushButton('Minimum Variance')
        self.minVarButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        builderLayout.addWidget(self.minVarButton)


        leftLayout.addWidget(builderFrame)
        leftLayout.addStretch()
        
        # history
        hFrame = QtWidgets.QGroupBox('Matrix History')
        hLayout = QtWidgets.QVBoxLayout(hFrame)

        mGrid = QtWidgets.QGridLayout()
        self.M = [] # matrix that is displayed in history
        for y in range(3):
            row = []
            for x in range(3):
                label = QtGui.QLabel('0.0')
                #label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                mGrid.addWidget(label, y, x, 1, 1)
                row.append(label)
            self.M.append(row)

        hLayout.addLayout(mGrid)

        hBotGrid = QtWidgets.QGroupBox()
        hBotLayout = QtWidgets.QHBoxLayout(hBotGrid)
        leftButtons = QtWidgets.QVBoxLayout()
        #loadMat = QtWidgets.QPushButton('Load Matrix')
        self.removeRow = QtWidgets.QPushButton('Remove Matrix')
        #leftButtons.addWidget(loadMat)
        leftButtons.addWidget(self.removeRow)
        leftButtons.addStretch()
        hBotLayout.addLayout(leftButtons,1)
        self.history = QtWidgets.QListWidget()
        self.history.setEditTriggers(QtWidgets.QAbstractItemView.SelectedClicked)
        hBotLayout.addWidget(self.history,2)
        hLayout.addWidget(hBotGrid)

        rightLayout.addWidget(hFrame)

        mainLayout.addStretch()

        ## bottom area with apply button
        #bottomLayout = QtGui.QHBoxLayout()
        #self.apply = QtGui.QPushButton('Apply')
        #self.apply.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        #bottomLayout.addWidget(self.apply)
        #applyLabel = QtGui.QLabel('Multiplies each data vector by (selected history matrix multiplied by rotation matrix)')
        #bottomLayout.addWidget(applyLabel)
        #mainLayout.addLayout(bottomLayout)
        

class ManRotUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Manual Rotation')
        Frame.resize(300,300)

        self.layout = QtWidgets.QVBoxLayout(Frame)

                # matrix A setup
        self.R = [] # current rotation matrix
        rFrame = QtWidgets.QGroupBox('Rotation Matrix')
        rLayout = QtWidgets.QGridLayout(rFrame)
        for y in range(3):
            row = []
            for x in range(3):
                edit = QtGui.QLineEdit()
                edit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly) #i dont even know if this does anything
                edit.setText('0.0')
                rLayout.addWidget(edit, y, x, 1, 1)
                row.append(edit)
            self.R.append(row)

        self.layout.addWidget(rFrame)

        extraButtons = QtWidgets.QHBoxLayout()
        self.loadIdentity = QtGui.QPushButton('Load Identity')
        self.loadIdentity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadZeros = QtGui.QPushButton('Load Zeros')
        self.loadZeros.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.loadIdentity)
        extraButtons.addWidget(self.loadZeros)
        extraButtons.addStretch()
        self.layout.addLayout(extraButtons)

        axFrame = QtWidgets.QGroupBox('By Axis Angle')
        axLayout = QtWidgets.QHBoxLayout(axFrame)
        angleLabel = QtGui.QLabel('Angle')
        angleLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.axisAngle = QtGui.QDoubleSpinBox()
        self.axisAngle.setWrapping(True)
        self.axisAngle.setMaximum(360.0)
        self.axisAngle.setSuffix('\u00B0')
        self.axisAngle.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        axLayout.addWidget(angleLabel)
        axLayout.addWidget(self.axisAngle)
        self.genButtons = []
        for ax in Mth.AXES:
            gb = QtGui.QPushButton(f'{ax}')
            gb.setMinimumWidth(5)
            gb.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
            axLayout.addWidget(gb)
            self.genButtons.append(gb)

        self.layout.addWidget(axFrame)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)



class MinVarUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Minimum Variance')
        Frame.resize(300,150)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        # setup xyz vector dropdowns
        vectorLayout = QtWidgets.QHBoxLayout()
        self.vector = []
        for i,ax in enumerate(Mth.AXES):
            v = QtWidgets.QComboBox()
            for dstr in window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    v.addItem(dstr)
            self.vector.append(v)
            vectorLayout.addWidget(v)
        vectorLayout.addStretch()

        self.layout.addLayout(vectorLayout)

        # setup datetime edits
        sliderFont = QtGui.QFont("monospace", 10 if window.OS == 'windows' else 14)#, QtGui.QFont.Bold) 
        self.startTimeEdit = QtWidgets.QDateTimeEdit()
        self.endTimeEdit = QtWidgets.QDateTimeEdit()
        self.startTimeEdit.setFont(sliderFont)
        self.endTimeEdit.setFont(sliderFont)
        self.startTimeEdit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.endTimeEdit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.startTimeEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        self.endTimeEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        minDateTime,maxDateTime = window.getMinAndMaxDateTime()
        self.startTimeEdit.setMinimumDateTime(minDateTime)
        self.startTimeEdit.setMaximumDateTime(maxDateTime)
        self.startTimeEdit.setDateTime(minDateTime)
        self.endTimeEdit.setMinimumDateTime(minDateTime)
        self.endTimeEdit.setMaximumDateTime(maxDateTime)
        self.endTimeEdit.setDateTime(maxDateTime)
        self.layout.addWidget(self.startTimeEdit)
        self.layout.addWidget(self.endTimeEdit)

        self.eigenValsLabel = QtWidgets.QLabel('')
        self.layout.addWidget(self.eigenValsLabel)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)
