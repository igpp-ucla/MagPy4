
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from fflib import ff_time
import numpy as np
import datetime, time
import bisect, functools
import os
import numpy.lib.recfunctions as rfn
from . import config

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
        self.viewEdtdDta = QtGui.QCheckBox('View Edited Data')
        self.timeVarLabel = QtWidgets.QLabel('Time Variable:')
        self.timeVarBox = QtWidgets.QComboBox()
        filePathLayout.addWidget(self.filePathCB)
        filePathLayout.addWidget(self.viewEdtdDta)
        filePathLayout.addWidget(self.timeVarLabel)
        filePathLayout.addWidget(self.timeVarBox)
        filePathLayout.addStretch()
        fileHL.addLayout(filePathLayout, 1, 1, 1, 1)
        topHL.addLayout(fileHL)
        topHL.addStretch()
        self.verticalLayout_2.addLayout(topHL)

        self.verticalLayout.addWidget(self.widget)
        self.dataTableView = QtGui.QTableView(dataFrame)
        self.dataTableView.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily(config.fonts['monospace'][0])
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
        self.dataTableView.setShowGrid(True)
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

        dataFrame.setWindowTitle("Flatfile Data")
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
        # Remove any trailing zeros that aren't necessary
        splitStr = UTC.split('.')
        if len(splitStr) > 1 and len(splitStr[1]) > 3 and splitStr[1][-1] == '0':
            UTC = UTC[:-1]

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
    def __init__(self, datain, parent=None, epoch=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.arraydata = datain
        self.headers = list(datain.dtype.names)
        self.headers = [hdr.strip(' ').strip('\n') for hdr in self.headers]
        self.epoch = 'Y2000'
        self.UTCMode = True
        if epoch:
            self.epoch = epoch
        self.nColumns = len(datain[0])
        self.nRows = len(self.arraydata)

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
            t = self.arraydata[index.row()][index.column()]
            if self.UTCMode:
                value = ff_time.tick_to_ts(t, self.epoch)
            else:
                value = "%16.4f" % t
        else:
            d = self.arraydata[index.row()][index.column()]
            value = str(d)
        return value

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignLeft
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal: # column labels
                return self.headers[section]
            else: # row labels
                # We add 1 so that row numbers start with 1 rather than 0
                return section + 1
        return None

    def tableDetailHeader(self, col, orientation, role): # i think this is unused
        print("TableDetailHeader")
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.headers[col]
        return "section"

class DataDisplay(QtGui.QFrame, DataDisplayUI):
    """ file data dialog """
    def __init__(self, window, FIDs, Title=None, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.window = window
        self.ui = DataDisplayUI()
        self.ui.setupUi(self)
        self.title = Title
        self.FIDs = FIDs
        self.curFID = FIDs[0]
        self.updateFile()
        self.table = None
        self.rangeSelection = None

        self.setFileComboNames()
        self.ui.fileCombo.currentIndexChanged.connect(self.fileComboChanged)
        self.ui.filePathCB.stateChanged.connect(self.setFileComboNames)
        self.ui.viewEdtdDta.stateChanged.connect(self.edtdDtaMode)

        self.ui.buttonBox.rejected.connect(self.close)
        buttonBox = self.ui.buttonBox
        saveButton = QtGui.QPushButton("&Save All Data")
        saveButton.clicked.connect(self.saveData)
        buttonBox.addButton(saveButton, QtGui.QDialogButtonBox.ApplyRole)
        saveRangeBtn = QtGui.QPushButton("Save Range of Data")
        saveRangeBtn.clicked.connect(self.selectRange)
        buttonBox.addButton(saveRangeBtn, QtGui.QDialogButtonBox.ActionRole)
        self.ui.checkBox.clicked.connect(self.toggleTimeDisplay)
        self.ui.moveByTime.clicked.connect(self.moveByTime)
        self.ui.moveByRow.clicked.connect(self.moveByRow)

        self.updateTimeVarBox()
        self.update()
        self.clip = QtGui.QApplication.clipboard()

        # Disable 'View Edited Data' checkbox if set to view only unedited data
        if self.window.currentEdit == 0:
            self.ui.viewEdtdDta.setEnabled(False)
        else:
            self.ui.viewEdtdDta.setEnabled(True)

        self.ui.timeVarBox.currentIndexChanged.connect(self.epochChanged)

    def closeEvent(self, event):
        self.closeRangeSelection()

    def selectRange(self):
        self.rangeSelection = RangeSelection(self)
        self.rangeSelection.show()

    def closeRangeSelection(self):
        if self.rangeSelection:
            self.rangeSelection.close()

    def fileComboChanged(self,i):
        self.curFID = self.FIDs[i]
        self.updateTimeVarBox()
        self.updateFile()
        self.update()
    
    def updateTimeVarBox(self):
        ''' Update time variable option box when file is changed '''
        # Clear previous combo box options
        showTimeVar = False
        self.ui.timeVarBox.clear()

        # If file is a CDF and more than one epoch is loaded,
        # display timeVarBox and add epoch variables to box
        if self.curFID.getFileType() == 'CDF':
            showTimeVar = True
            timeVars = self.curFID.getEpochVars()
            if len(timeVars) <= 1:
                showTimeVar = False
            self.ui.timeVarBox.addItems(timeVars)

        # Hide/show timeVarBox and its label
        self.ui.timeVarBox.setVisible(showTimeVar)
        self.ui.timeVarLabel.setVisible(showTimeVar)

    def setFileComboNames(self):
        self.ui.fileCombo.blockSignals(True)
        self.ui.fileCombo.clear()
        fullPath = self.ui.filePathCB.isChecked()
        for FID in self.FIDs:
            name = FID.name if fullPath else os.path.split(FID.name)[1]
            self.ui.fileCombo.addItem(name)
        self.ui.fileCombo.adjustSize()
        self.ui.fileCombo.blockSignals(False)

    def epochChanged(self):
        self.updateFile()
        self.update()

    def updateFile(self):
        timeVar = None
        if self.ui.timeVarBox.currentText() != '':
            timeVar = self.ui.timeVarBox.currentText()
        
        if self.curFID.getFileType() == 'ASCII':
            timeVar = self.window.epoch

        self.dataByRec = self.curFID.getRecords(timeVar)
        time_key = self.dataByRec.dtype.names[0]
        self.time = np.array(self.dataByRec[time_key])

    def edtdDtaMode(self):
        if self.ui.viewEdtdDta.isChecked():
            # If user manages to check the view edited data button while
            # the original data is being displayed, reset window settings
            if self.window.currentEdit == 0:
                self.ui.viewEdtdDta.setChecked(False)
                self.ui.viewEdtdDta.setEnabled(False)
                self.edtdDtaMode()
                return
            # Otherwise, grey out filepath combo-box and checkboxes
            self.ui.filePathCB.setEnabled(False)
            self.ui.fileCombo.setEnabled(False)
        else:
            # Re-enable file chooser and update table data with FF data
            self.ui.filePathCB.setEnabled(True)
            self.ui.fileCombo.setEnabled(True)
            self.updateFile()
        self.update()

    # Used to find list of times with the longest length
    def findLargestTimeRng(self, tms):
        maxIndex = 0
        i = 0
        for tm in tms:
            if len(tm) > len(tms[maxIndex]):
                maxIndex = i
            i = i + 1
        return tms[maxIndex]

    # Given a list of times and a min/max time from another list of times,
    # try to find the indices where these min/max times occur in the list
    def findStartEndIndices(self, timeVals, minTime, maxTime):
        # Find index where second list starts in first list
        startIndex = 0
        for t in timeVals:
            if t == minTime:
                break
            startIndex += 1

        # Find index where second list ends in first list
        timeValsLen = len(timeVals)
        endIndex = startIndex
        for i in range(startIndex, timeValsLen):
            if timeVals[i] == maxTime:
                break
            endIndex += 1

        return startIndex, endIndex

    def getEditedData(self):
        dta = []
        tms = []
        hdrs = ['SCET']

        for dstr in self.window.DATADICT.keys():
            dataValsList = self.window.DATADICT[dstr]
            en = self.window.currentEdit

            # If dstr has no value for current edit:
            if dataValsList[en] == []:
                # Look for the most recent edit number that has data for it
                while dataValsList[en] == [] and en >= 0:
                    en = en - 1

            # If only unedited data available, skip this datastring
            if en == 0:
                continue

            # Get the matching time values list and dataVals
            timeVals, res, avgRes = self.window.getTimes(dstr, en)
            dataVals = dataValsList[en]

            # Add dataVals, timeVals, and the datastring to overall lists
            dta.append(list(dataVals))
            tms.append(list(timeVals))
            hdrs.append(dstr)

        if self.window.insightMode and list(self.window.changeLog.keys()) != []:
            changeLogDta, changeLogTimes = self.createChangeDta()
            dta.append(changeLogDta)
            tms.append(changeLogTimes)
            hdrs.append('Changes')

        # Get the list of time values with the largest range
        timeLine = self.findLargestTimeRng(tms)
        i = 0
        for tvs in tms:
            s = tvs[0]  # First time in table
            e = tvs[-1] # Last time in table
            start, end = self.findStartEndIndices(timeLine, s, e)
            # For each dstr's values list, if there is no value at a given
            # time, fill it with NaN
            ppnd = [np.nan]*start
            apnd = [np.nan]*(len(timeLine) - end - 1)
            dta[i] = ppnd + dta[i] + apnd
            i = i+1
        return (dta, timeLine, hdrs)     

    def createChangeDta(self):
        changeKeys = list(self.window.changeLog.keys())
        times = self.window.getTimes(changeKeys[0], self.window.currentEdit)[0]
        dta = np.zeros(len(times))
        for dstr in changeKeys:
            lst = self.window.changeLog[dstr]
            for strt, end in lst:
                strt = self.window.calcDataIndexByTime(times, strt)
                end = self.window.calcDataIndexByTime(times, end)
                dta[strt:end] = [1]*(end-strt)
        return list(dta), list(times)

    def updtTimeEditAndStats(self, nRows, nCols, epoch):
        # Update time edit parameters with UTC strings
        start = ff_time.tick_to_date(self.time[0], epoch)
        stop_ = ff_time.tick_to_date(self.time[-1], epoch)
        self.ui.dateTimeEdit.setDateTime(start)
        self.ui.dateTimeEdit.setMinimumDateTime(start)
        self.ui.dateTimeEdit.setMaximumDateTime(stop_) 
        
        # Update statistics in corner of window
        nrows = 'Rows: ' + str(nRows)
        ncols = 'Columns: ' + str(nCols)
        epoch = 'Epoch: ' + str(epoch)
        startTime = start
        endTime = stop_
        stats = f'{nrows}, {ncols}, {epoch}          \n{startTime}\n{endTime}'
        self.ui.timesLabel.setText(stats)    

    def createTable(self, header, epoch):
        if self.dataByRec is not None:        
            tm = FFTableModel(self.dataByRec, parent=None, epoch=epoch)
            tm.setUTC(not self.ui.checkBox.isChecked())
            self.ui.dataTableView.setModel(tm)
            self.ui.dataTableView.resizeColumnToContents(0) # make time column resize to fit        

    def updateEditedData(self):
        self.setWindowTitle('Edited Data')

        # Get edited data and update class info/data
        dta, tms, hdrs = self.getEditedData()

        # Reformat data into a structured numpy array
        records = np.hstack([np.vstack(tms), np.transpose(dta)])
        dtype = np.dtype([(label, 'f4') for label in hdrs])
        records = rfn.unstructured_to_structured(records, dtype=dtype)

        self.dataByRec = records
        self.time = records[hdrs[0]]
        self.headerStrings = hdrs

        header = [h for h in hdrs]

        # Set table row settings
        self.ui.Row.setMinimum(1)
        self.ui.Row.setMaximum(len(tms))

        # Create table from data
        epoch = self.window.epoch
        self.createTable(header, epoch)
        self.updtTimeEditAndStats(len(tms), len(dta)+1, epoch)

    def updateFfData(self):
        self.setWindowTitle(self.title)
        if len(self.time) == 0:
            return

        # Update row settings in table
        rO = 1
        rE = len(self.dataByRec)
        self.ui.Row.setMinimum(rO)
        self.ui.Row.setMaximum(rE)

        # Create table from flatfile data
        epoch = self.curFID.getEpoch()
        if epoch is None:
            epoch = self.window.epoch
        self.createTable([], epoch)
        self.updtTimeEditAndStats(len(self.time), len(self.dataByRec[0]), epoch)           

    def update(self):
        if len(self.time) == 0:
            return

        if self.ui.viewEdtdDta.isChecked():
            self.updateEditedData()
        else:
            self.updateFfData()
            epoch = self.curFID.getEpoch()
            if epoch is None:
                epoch = self.window.epoch
            start = ff_time.tick_to_ts(self.time[0], epoch)
            self.ui.dateTimeEdit.setDateTime(UTCQDate.UTC2QDateTime(start))        

    # saves flatfile data to a plain text file
    def saveData(self, sig, indices=None):
        defaultSfx = '.csv'
        QQ = QtGui.QFileDialog(self)
        QQ.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        path = os.path.expanduser(".")
        QQ.setDirectory(path)
        fullname = QQ.getSaveFileName(parent=None, directory=path, caption="Save Data", filter='CSV file (*.csv)')
        if fullname is None:
            print('Save failed')
            return
        if fullname[0] == '':
            print('Save cancelled')
            return

        # Shape data
        selectedDta = self.dataByRec
        if indices is not None:
            startIndex, endIndex = indices
            selectedDta = selectedDta[startIndex:endIndex+1]

        # Create header to write at top of file
        header = ','.join(list(selectedDta.dtype.names))

        # If file name doesn't end with default suffix, add it before saving
        filename = fullname[0]
        if filename.endswith(defaultSfx) == False:
            filename += defaultSfx

        # Write data to file
        np.savetxt(filename, selectedDta, fmt = '%.10f', header=header, 
            delimiter=',', comments='')

    def toggleTimeDisplay(self):
        self.update()

    def moveByTime(self):
        row = self.getRowFrmDTime(self.ui.dateTimeEdit)
        self.ui.dataTableView.selectRow(row)

    def getRowFrmDTime(self, dt):
        formMM = 'yyyy MM dd hh mm ss zzz'
        formMMM = 'yyyy MMM dd hh:mm:ss.zzz'
        DT = dt.dateTime().toPyDateTime()
        epoch = self.curFID.getEpoch()
        if epoch is None:
            epoch = self.window.epoch
        t = ff_time.date_to_tick(DT, epoch)
        iRow = 0
        iRow = bisect.bisect(self.time, t)
        return iRow

    def moveByRow(self):
        # grab data ui.Row
        row = self.ui.Row.value() - 1
        self.ui.dataTableView.selectRow(row)

class RangeSelectionUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle("Range Selection")
        Frame.resize(200, 150)

        layout = QtWidgets.QGridLayout(Frame)
        instrLbl = QtWidgets.QLabel('Please select a range of data to save.')

        # Set up time range selection
        timeEditLayout = QtWidgets.QHBoxLayout()
        self.startTimeEdit = QtGui.QDateTimeEdit()
        startLbl = QtGui.QLabel('Start time: ')
        startRwLbl = QtWidgets.QLabel(' Start row: ')
        self.startRwBox = QtWidgets.QSpinBox()
        for e in [startLbl, self.startTimeEdit, startRwLbl, self.startRwBox]:
            timeEditLayout.addWidget(e)
        timeEditLayout.addStretch()

        # Set up row range selection
        rowLayout = QtWidgets.QHBoxLayout()
        self.endTimeEdit = QtGui.QDateTimeEdit()
        endLbl = QtGui.QLabel('End time:   ')
        self.endRwBox = QtWidgets.QSpinBox()
        endRwLbl = QtWidgets.QLabel(' End row:   ')
        for e in [endLbl, self.endTimeEdit, endRwLbl, self.endRwBox]:
            rowLayout.addWidget(e)
        rowLayout.addStretch()

        # Set up min/max and default values for range selectors
        start = ff_time.tick_to_date(window.time[0], window.window.epoch)
        stop_ = ff_time.tick_to_date(window.time[-1], window.window.epoch)
        self.startTimeEdit.setDateTime(start)
        self.endTimeEdit.setDateTime(stop_)
        for te in [self.startTimeEdit, self.endTimeEdit]:
            te.setDisplayFormat('yyyy MMM dd hh:mm:ss')
            te.setMinimumDateTime(start)
            te.setMaximumDateTime(stop_)

        maxRows = window.ui.Row.maximum()
        for rowObj in [self.startRwBox, self.endRwBox]:
            rowObj.setMinimum(1)
            rowObj.setMaximum(maxRows)
        self.endRwBox.setValue(maxRows)

        # Save button
        self.saveBtn = QtWidgets.QPushButton('Save Data')
        self.saveBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Add everything to layout
        layout.addWidget(instrLbl, 0, 0, 1, 2)
        layout.addLayout(timeEditLayout, 1, 0, 1, 2)
        layout.addLayout(rowLayout, 2, 0, 1, 2)
        layout.addWidget(self.saveBtn, 4, 1, 1, 1)

class RangeSelection(QtWidgets.QFrame, RangeSelectionUI):
    def __init__(self, window):
        super(RangeSelection, self).__init__(None)
        self.window = window
        self.ui = RangeSelectionUI()
        self.ui.setupUI(self, window)

        # Set up UI element connections
        self.ui.startTimeEdit.dateTimeChanged.connect(functools.partial(self.updateRowByTime, self.ui.startTimeEdit, self.ui.startRwBox))
        self.ui.endTimeEdit.dateTimeChanged.connect(functools.partial(self.updateRowByTime, self.ui.endTimeEdit, self.ui.endRwBox))
        self.ui.startRwBox.valueChanged.connect(functools.partial(self.updateTimeEditByRow, self.ui.startTimeEdit))
        self.ui.endRwBox.valueChanged.connect(functools.partial(self.updateTimeEditByRow, self.ui.endTimeEdit))
        self.ui.saveBtn.clicked.connect(self.saveRangeData)

    def updateRowByTime(self, te, rbox):
        # Get row number from current time
        row = self.window.getRowFrmDTime(te) + 1
        # Set row spinbox value without signals
        rbox.blockSignals(True)
        rbox.setValue(row)
        rbox.blockSignals(False)

    def updateTimeEditByRow(self, te, row):
        row = row-1
        row = max(0, row)
        # Get tick from data times corresponding to row number
        newTime = self.window.time[row]
        # Get UTC version of time and update timeEdit
        date = ff_time.tick_to_date(newTime, self.window.window.epoch)
        te.blockSignals(True)
        te.setDateTime(date)
        te.blockSignals(False)

    def saveRangeData(self):
        indices = (self.ui.startRwBox.value()-1, self.ui.endRwBox.value()-1)
        self.window.saveData(True, indices)