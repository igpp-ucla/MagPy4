
from PyQt5 import QtGui, QtCore, QtWidgets
from FF_Time import FFTIME, FF_EPOCH
import datetime
import numpy as np
import os

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class DataDisplayUI(object):
    def setupUi(self, dataFrame):
        dataFrame.setObjectName(_fromUtf8("dataFrame"))
        dataFrame.resize(1106, 790)
        dataFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        dataFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(dataFrame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.widget = QtGui.QWidget(dataFrame)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.widget)
        #self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.fileLabel = QtGui.QLabel(self.widget)
        self.fileLabel.setObjectName(_fromUtf8("fileLabel"))
        self.verticalLayout_2.addWidget(self.fileLabel)
        self.timesLabel = QtGui.QLabel(self.widget)
        self.timesLabel.setObjectName(_fromUtf8("timesLabel"))
        self.verticalLayout_2.addWidget(self.timesLabel)
        self.verticalLayout.addWidget(self.widget)
        self.dataTableView = QtGui.QTableView(dataFrame)
        self.dataTableView.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Monospace"))
        self.dataTableView.setFont(font)
        self.dataTableView.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dataTableView.setTabKeyNavigation(False)
        self.dataTableView.setProperty("showDropIndicator", False)
        self.dataTableView.setDragDropOverwriteMode(False)
        self.dataTableView.setWordWrap(False)
        self.dataTableView.setCornerButtonEnabled(False)
        self.dataTableView.setObjectName(_fromUtf8("dataTableView"))
        self.dataTableView.horizontalHeader().setHighlightSections(False)
        self.dataTableView.verticalHeader().setHighlightSections(False)
        self.verticalLayout.addWidget(self.dataTableView)
        self.widget_2 = QtGui.QWidget(dataFrame)
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget_2)
        #self.horizontalLayout.setMargin(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.checkBox = QtGui.QCheckBox(self.widget_2)
        self.checkBox.setObjectName(_fromUtf8("checkBox"))
        self.horizontalLayout.addWidget(self.checkBox)
        self.moveByTime = QtGui.QPushButton(self.widget_2)
        self.moveByTime.setObjectName(_fromUtf8("moveByTime"))
        self.horizontalLayout.addWidget(self.moveByTime)
        self.dateTimeEdit = QtGui.QDateTimeEdit(self.widget_2)
        self.dateTimeEdit.setObjectName(_fromUtf8("dateTimeEdit"))
        self.horizontalLayout.addWidget(self.dateTimeEdit)
        self.moveByRow = QtGui.QPushButton(self.widget_2)
        self.moveByRow.setObjectName(_fromUtf8("moveByRow"))
        self.horizontalLayout.addWidget(self.moveByRow)
        self.Row = QtGui.QSpinBox(self.widget_2)
        self.Row.setObjectName(_fromUtf8("Row"))
        self.horizontalLayout.addWidget(self.Row)
        self.verticalLayout.addWidget(self.widget_2)
        self.buttonBox = QtGui.QDialogButtonBox(dataFrame)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Close)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        #self.retranslateUi(dataFrame)
        dataFrame.setWindowTitle("Flat File Data")
        self.fileLabel.setText("TextLabel")
        self.timesLabel.setText("TextLabel")
        self.checkBox.setText("Time Ticks")
        self.moveByTime.setText("Go to Time")
        self.moveByRow.setText("Go to Row")

        QtCore.QMetaObject.connectSlotsByName(dataFrame)

class FFTableModel(QtCore.QAbstractTableModel):
    def __init__(self, time_in, datain, headerdata, parent=None, epoch=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.time = time_in
        self.arraydata = datain
        self.headerdata = headerdata
        self.epoch = FF_EPOCH.Y2000
        self.UTCMode = True
        if epoch:
            self.epoch = epoch
        self.nColumns = len(datain) + 1
        self.nRows = len(time_in)

    def rowCount(self, parent):
        return self.nRows

    def columnCount(self, parent):
        return self.nColumns

    def setUTC(self, utc):
        self.UTCMode = utc

    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None
        if index.column() == 0:
            t = self.time[index.row()]
            if self.UTCMode:
                if isinstance(t, FFTIME):
                    utc = t.UTC
                else:
                    utc = FFTIME(t, Epoch=self.epoch).UTC
                utc = UTCQDate.removeDOY(utc)
                value = utc
            else:
                value = "%16.4f" % t
        else:
            d = self.arraydata[index.column()-1][index.row()]
            value = str(d)
        return value


    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return (self.headerdata[section])
            else:
                return (section + 1)
        return None

    def tableDetailHeader(self, col, orientation, role):
        print("TableDetailHeader")
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return (self.headerdata[col])
        return "section"


class UTCQDate():
    FORMAT = "yyyy MMM  d hh:mm:ss.zzz"
    FORMAT2 = "yyyy MMM dd hh:mm:ss.zzz"
    FORMAT3 = "yyyy MMM dd  hh:mm:ss.zzz"

    def removeDOY(UTC):
        cut = UTC[4:8]
        return UTC.replace(cut, '').strip()

    # convert UTC string to QDateTime
    def UTC2QDateTime(UTC):
        UTC = UTCQDate.removeDOY(UTC)
        qdateTime = QtCore.QDateTime.fromString(UTC, UTCQDate.FORMAT)
        test = qdateTime.toString()
        if not test:
            qdateTime = QtCore.QDateTime.fromString(UTC, UTCQDate.FORMAT2)
        test = qdateTime.toString()
        if not test:
            qdateTime = QtCore.QDateTime.fromString(UTC, UTCQDate.FORMAT3)
        #test = qdateTime.toString()
        #UTCQDate.doy(qdateTime)
        return qdateTime

class DataDisplay(QtGui.QFrame, DataDisplayUI):
    """ file data dialog """
    def __init__(self, FID, time, data, Title=None, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = DataDisplayUI()
        self.ui.setupUi(self)
        self.title = Title
        self.FID = FID
        self.time = time
        self.data = data
        self.setActions()
        self.update()

    def update(self):
        parm = self.FID.FFParm
        info = self.FID.FFInfo
        self.ui.fileLabel.setText(parm["DATA"].info)
        self.setWindowTitle(self.title)

        startTime = info["FIRST_TIME"].info.replace("UTC", '')
        endTime = info["LAST_TIME"].info.replace("UTC", '')
        epoch = " Epoch " + parm["EPOCH"].info
        nrows = "NROWS : " + parm["NROWS"].info.lstrip()
        ncols = "NCOLS : " + parm["NCOLS"].info.lstrip()
        stats = f'Times: [{startTime}]>>[{endTime}] [{epoch}] [{nrows}] [{ncols}]'
        self.ui.timesLabel.setText(stats)
        epoch = self.FID.getEpoch()
        # actually it's the last data (data not allfile)
        nRow = self.FID.getRows()
        rO = self.FID.ffsearch(self.time[0], 1, nRow)
        rE = self.FID.ffsearch(self.time[-1], 1, nRow)
        if rE is None:
            rE = nRow
        self.ui.Row.setMinimum(rO)
        self.ui.Row.setMaximum(rE)
        start = FFTIME(self.time[0], Epoch=epoch)
        stop_ = FFTIME(self.time[-1], Epoch=epoch)
        self.ui.dateTimeEdit.setDateTime(UTCQDate.UTC2QDateTime(start.UTC))
        self.ui.dateTimeEdit.setMinimumDateTime(UTCQDate.UTC2QDateTime(start.UTC))
        self.ui.dateTimeEdit.setMaximumDateTime(UTCQDate.UTC2QDateTime(stop_.UTC))
        header = self.FID.getColumnDescriptor("NAME")
        if self.data is not None:
            tm = FFTableModel(self.time, self.data, header, parent=None, epoch=parm["EPOCH"].value)
            tm.setUTC(not self.ui.checkBox.isChecked())
            self.ui.dataTableView.setModel(tm)
            self.ui.dataTableView.resizeColumnToContents(0) # make time column resize to fit

    def writeText(self, fullname):
        np.set_printoptions(formatter={'float': '{:+10.4f}'.format})
        file = open(fullname[0], "+w")
        t = self.time
        d = self.data
        epoch = self.FID.FFParm["EPOCH"].value
        nRows = len(self.time)
        for i in range(nRows):
            UTC = FFTIME(t[i], Epoch=epoch).UTC
            file.write(UTC + " " + str(d[:, i])[1:-2] + "\n")
        file.close()
        return

    def saveData(self):
        QQ = QtGui.QFileDialog(self)
        QQ.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        path = os.path.expanduser("~")
        QQ.setDirectory(path)
        fullname = QQ.getSaveFileName(parent=None, directory=path, caption="Save Data")
        if fullname is not None:
            self.writeText(fullname)
            print(f'{fullname} saved')

    def toggleTimeDisplay(self):
        self.update()

    def moveByTime(self):
        # grab data ui.dateTimeEdit
        formMM = "yyyy MM dd hh mm ss zzz"
        formMMM = "yyyy MMM dd hh:mm:ss.zzz"
        DT = self.ui.dateTimeEdit.dateTime()
        DTSTR = DT.toString(formMM)
        DTSTRM = DT.toString(formMMM)
        dtList = DTSTR.split()
        dtInt = [int(item) for item in dtList]
        doy = datetime.datetime(*dtInt).timetuple().tm_yday
        UTC = dtList[0] + " " + str(doy) + DTSTRM[4:]
        #print(UTC)
        t = FFTIME(UTC, Epoch=self.FID.getEpoch()).tick
        iRow = self.FID.ffsearch(t, 1, self.FID.getRows()) - 1
        self.ui.dataTableView.selectRow(iRow)

    def moveByRow(self):
        # grab data ui.Row
        row = self.ui.Row.value() - 1
        self.ui.dataTableView.selectRow(row)

    def setActions(self):
        self.ui.buttonBox.rejected.connect(self.close)
        buttonBox = self.ui.buttonBox
        saveButton = QtGui.QPushButton("Save Data")
        saveButton.clicked.connect(self.saveData)
        buttonBox.addButton(saveButton, QtGui.QDialogButtonBox.ApplyRole)
        self.ui.checkBox.clicked.connect(self.toggleTimeDisplay)
        self.ui.moveByTime.clicked.connect(self.moveByTime)
        self.ui.moveByRow.clicked.connect(self.moveByRow)
#       printButton = QPushButton("Print Data")
#       printPanelButton.clicked.connect(self.printPanel)
#       buttonBox.addButton(printButton, QDialogButtonBox.ApplyRole)



