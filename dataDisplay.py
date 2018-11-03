
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import FF_File
from FF_Time import FFTIME, FF_EPOCH
import numpy as np
import datetime, time
import os
# from tqdm import tqdm

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class DataDisplayUI(object):
    def setupUi(self, dataFrame):
        dataFrame.setObjectName(_fromUtf8("dataFrame"))
        dataFrame.resize(1000, 700)
        dataFrame.setFrameShape(QtGui.QFrame.StyledPanel)
        dataFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.verticalLayout = QtGui.QVBoxLayout(dataFrame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.widget = QtGui.QWidget(dataFrame)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))

        topHL = QtGui.QHBoxLayout()
        self.timesLabel = QtGui.QLabel(self.widget)
        self.timesLabel.setObjectName(_fromUtf8("timesLabel"))
        self.timesLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        topHL.addWidget(self.timesLabel)
        fileHL = QtGui.QGridLayout()
        fileHL.addWidget(QtGui.QLabel('File Name:'), 0, 0, 1, 1)
        self.fileCombo = QtGui.QComboBox(self.widget)
        self.fileCombo.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.fileCombo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)
        fileHL.addWidget(self.fileCombo, 0, 1, 1, 1)
        filePathLayout = QtGui.QHBoxLayout()
        self.filePathCB = QtGui.QCheckBox('Show Full &Path')
        filePathLayout.addWidget(self.filePathCB)
        filePathLayout.addStretch()
        fileHL.addLayout(filePathLayout, 1, 1, 1, 1)
        topHL.addLayout(fileHL)
        topHL.addStretch()
        self.verticalLayout_2.addLayout(topHL)

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

        dataFrame.setWindowTitle("Flat File Data")
        self.timesLabel.setText("TextLabel")
        self.checkBox.setText("Sho&w Ticks")
        self.moveByTime.setText("Go to &Time")
        self.moveByRow.setText("Go to &Row")

        QtCore.QMetaObject.connectSlotsByName(dataFrame)

class UTCQDate():
    FORMAT = "yyyy MMM  d hh:mm:ss.zzz"
    FORMAT2 = "yyyy MMM dd hh:mm:ss.zzz"
    FORMAT3 = "yyyy MMM dd  hh:mm:ss.zzz"

    def removeDOY(UTC): # should just add this as option to ffPy?
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
        return qdateTime

    def QDateTime2UTC(qdt):
        doy = QtCore.QDateTime.fromString(f'{qdt.date().year()} 01 01', 'yyyy MM dd').daysTo(qdt) + 1
        DOY = "%03d" % doy
        dateTime = qdt.toString(UTCQDate.FORMAT)
        return dateTime[:5] + DOY + dateTime[4:]
       

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
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignLeft
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal: # column labels
                return self.headerdata[section]
            else: # row labels
                return section
        return None

    def tableDetailHeader(self, col, orientation, role): # i think this is unused
        print("TableDetailHeader")
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.headerdata[col]
        return "section"

class DataDisplay(QtGui.QFrame, DataDisplayUI):
    """ file data dialog """
    def __init__(self, FIDs, Title=None, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = DataDisplayUI()
        self.ui.setupUi(self)
        self.title = Title
        self.FIDs = FIDs
        self.curFID = FIDs[0]
        self.updateFile()

        self.setFileComboNames()
        self.ui.fileCombo.currentIndexChanged.connect(self.fileComboChanged)
        self.ui.filePathCB.stateChanged.connect(self.setFileComboNames)

        self.ui.buttonBox.rejected.connect(self.close)
        buttonBox = self.ui.buttonBox
        saveButton = QtGui.QPushButton("&Save Data")
        saveButton.clicked.connect(self.saveData)
        buttonBox.addButton(saveButton, QtGui.QDialogButtonBox.ApplyRole)
        self.ui.checkBox.clicked.connect(self.toggleTimeDisplay)
        self.ui.moveByTime.clicked.connect(self.moveByTime)
        self.ui.moveByRow.clicked.connect(self.moveByRow)
#       printButton = QPushButton("Print Data")
#       printPanelButton.clicked.connect(self.printPanel)
#       buttonBox.addButton(printButton, QDialogButtonBox.ApplyRole)


        self.update()
        self.clip = QtGui.QApplication.clipboard()

    def fileComboChanged(self,i):
        self.curFID = self.FIDs[i]
        self.updateFile()
        self.update()

    def setFileComboNames(self):
        self.ui.fileCombo.blockSignals(True)
        self.ui.fileCombo.clear()
        fullPath = self.ui.filePathCB.isChecked()
        for FID in self.FIDs:
            name = FID.name if fullPath else os.path.split(FID.name)[1]
            self.ui.fileCombo.addItem(name)
        #self.ui.fileCombo.resize(self.ui.fileCombo.sizeHint())
        #print(self.ui.fileCombo.sizeHint())
        self.ui.fileCombo.adjustSize()
        self.ui.fileCombo.blockSignals(False)


    def updateFile(self):
        nRows = self.curFID.getRows()
        records = self.curFID.DID.sliceArray(row=1, nRow=nRows)
        self.time = records["time"]
        self.dataByRec = records["data"]
        self.dataByCol = FF_File.arrayToColumns(records["data"])

    def update(self):
        parm = self.curFID.FFParm
        info = self.curFID.FFInfo
        #self.ui.fileLabel.setText(parm["DATA"].info)
        self.setWindowTitle(self.title)

        startTime = info["FIRST_TIME"].info.replace("UTC", '')
        endTime = info["LAST_TIME"].info.replace("UTC", '')
        epoch = "Epoch : " + parm["EPOCH"].info
        nrows = "NROWS : " + parm["NROWS"].info.lstrip()
        ncols = "NCOLS : " + parm["NCOLS"].info.lstrip()
        #stats = f'Times: [{startTime}]>>[{endTime}] [{epoch}] [{nrows}] [{ncols}]'
        stats = f'{nrows}, {ncols}, {epoch}          \n{startTime}\n{endTime}'
        self.ui.timesLabel.setText(stats)
        epoch = self.curFID.getEpoch()
        # actually it's the last data (data not allfile)
        nRow = self.curFID.getRows()
        rO = self.curFID.ffsearch(self.time[0], 1, nRow)
        rE = self.curFID.ffsearch(self.time[-1], 1, nRow)
        if rE is None:
            rE = nRow
        self.ui.Row.setMinimum(rO)
        self.ui.Row.setMaximum(rE)
        start = FFTIME(self.time[0], Epoch=epoch)
        stop_ = FFTIME(self.time[-1], Epoch=epoch)
        self.ui.dateTimeEdit.setDateTime(UTCQDate.UTC2QDateTime(start.UTC))
        self.ui.dateTimeEdit.setMinimumDateTime(UTCQDate.UTC2QDateTime(start.UTC))
        self.ui.dateTimeEdit.setMaximumDateTime(UTCQDate.UTC2QDateTime(stop_.UTC))
        self.headerStrings = self.curFID.getColumnDescriptor("NAME")
        units = self.curFID.getColumnDescriptor("UNITS")
        header = [f'{h} ({u})' if u else f'{h}' for h,u in zip(self.headerStrings,units)]
        if self.dataByCol is not None:
            tm = FFTableModel(self.time, self.dataByCol, header, parent=None, epoch=parm["EPOCH"].value)
            tm.setUTC(not self.ui.checkBox.isChecked())
            self.ui.dataTableView.setModel(tm)
            self.ui.dataTableView.resizeColumnToContents(0) # make time column resize to fit

    # saves flatfile data to a plain text file
    def saveData(self):
        QQ = QtGui.QFileDialog(self)
        QQ.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        path = os.path.expanduser(".")
        QQ.setDirectory(path)
        fullname = QQ.getSaveFileName(parent=None, directory=path, caption="Save Data", filter='Text files (*.txt)')
        if fullname is None:
            print('Save failed')
            return
        if fullname[0] == '':
            print('Save cancelled')
            return
        np.set_printoptions(formatter={'float': '{:+10.4f}'.format}, linewidth=10000)
        print(fullname)
        epoch = self.curFID.FFParm["EPOCH"].value
        nRows = len(self.time)

		#have option for 'readable format' vs 'excel ready'
        #print('reshaping data for printing...')

        shape = np.shape(self.dataByRec)
        dataShaped = np.zeros((shape[0], shape[1]+1))
        dataShaped[:,1:] = self.dataByRec
        dataShaped[:,0] = self.time
        print(f'Writing {nRows} records to {fullname[0]}...')
        np.savetxt(fullname[0], dataShaped, fmt = '%+10.4f')

        #file = open(fullname[0], "+w")
        
        #fileStr = ''
        #utcs = [UTCQDate.removeDOY(FFTIME(t, Epoch=epoch).UTC) for t in self.time]
        #fileStrs = [f'{utcs[i]} {str(self.dataByCol[:,1])[1:-2]}' for i in range(nRows)]

        #utc = FFTIME(self.time, Epoch=epoch).UTC
        #for i in tqdm(range(nRows),ascii=True):
            #UTC = FFTIME(self.time[i], Epoch=epoch).UTC
            #UTC = UTCQDate.removeDOY(UTC)

            #file.write(f'{UTC} {str(self.dataByCol[:,i])[1:-2]}\n')
            #fileStrs.append(f'{utcs} {str(self.dataByCol[:,i])[1:-2]}')

        #file.write('\n'.join(fileStrs))
        #file.write(fileStr)
        #file.close()

        print(f'Save complete')
        # should auto open the file here afterwards?



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
        t = FFTIME(UTC, Epoch=self.curFID.getEpoch()).tick
        iRow = self.curFID.ffsearch(t, 1, self.curFID.getRows()) - 1
        self.ui.dataTableView.selectRow(iRow)

    def moveByRow(self):
        # grab data ui.Row
        row = self.ui.Row.value() - 1
        self.ui.dataTableView.selectRow(row)

    # this doesn't work anymore. the key value looks like its just wrong?
    def keyPressEvent(self, e):
        #if (e.modifiers() & QtCore.Qt.ControlModifier):
        #    print(f'{QtCore.Qt.Key_C} {e.key()}')
        #    if e.key() == QtCore.Qt.Key_C: #copy
        if e.matches(QtGui.QKeySequence.Copy):
            selected = self.ui.dataTableView.selectedIndexes()
            if selected:
                # sort the indexes to get them in order and put them in string table
                rows = sorted(index.row() for index in selected)
                columns = sorted(index.column() for index in selected)
                rowcount = rows[-1] - rows[0] + 1
                colcount = columns[-1] - columns[0] + 1

                table = [[''] * colcount for _ in range(rowcount)]
                for index in selected:
                    row = index.row() - rows[0]
                    column = index.column() - columns[0]
                    table[row][column] = index.data()
                # build final string to copy to clipboard
                s = 'Record\t' + '\t'.join(self.headerStrings[columns[0]:columns[-1]+1])
                for r,row in enumerate(table):
                    s = f'{s}\n{rows[0]+r}\t'
                    s = s + '\t'.join(row)
                    s = s[:-1]
                print(s)
                self.clip.setText(s)

        e.accept()

