from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from FF_Time import FFTIME, leapFile
from datetime import datetime
from .layoutTools import BaseLayout
import numpy as np
from bisect import bisect
import time
import re
import numpy.lib.recfunctions as rfn

class TextFD():
    def __init__(self, filename, fileType, ancInfo):
        self.name = filename
        self.filename = filename
        self.fileType = fileType
        self.ancInfo = ancInfo

        self.epoch = self.ancInfo['Epoch']
        self.times = self.ancInfo['Times']
        self.labels = self.ancInfo['Labels']
        self.units = self.ancInfo['Units']
        self.records = self.ancInfo['Records']
        self.numRows = len(self.times)
        self.sources = []*len(self.units)
        self.stateInfo = self.ancInfo['StateInfo']

        self.fd = None

    def getFileType(self):
        return 'ASCII'

    def getName(self):
        return self.name

    def open(self):
        if self.fd is None:
            self.fd = open(self.filename, 'r')

    def getEpoch(self):
        return self.epoch

    def getUnits(self):
        return ['sec'] + self.units
    
    def getLabels(self):
        return ['SCET'] + self.labels
    
    def ffSearch(self, tick, startRow, endRow):
        startRow = min(startRow-1, 0) 
        endRow = max(endRow-1, self.numRows)
        return bisect(self.times[startRow:endRow], tick)

    def getRows(self):
        return len(self.times)

    def getRecords(self, epoch_var=None):
        # Add units to column labels
        labels = self.getLabels()
        units = self.getUnits()
        new_labels = []
        for label, unit in zip(labels, units):
            if unit:
                label = f'{label} ({unit})'
            new_labels.append(label)

        # Create array dtype
        dtype = [(label, 'f4') for label in new_labels]
        dtype = np.dtype(dtype)

        # Restructure data in a numpy records table with the times
        # set as the first column
        nRows = self.getRows()
        table = np.hstack([np.reshape(self.times, (nRows, 1)), self.records])
        datas = rfn.unstructured_to_structured(table, dtype=dtype)
        return datas

    def close(self):
        if self.fd:
            self.fd.close()
            self.fd = None

class Asc_UI(BaseLayout):
    def setupUI(self, window, frame):
        frame.resize(300, 100)
        frame.setWindowTitle('ASCII File Settings')
        layout = QtWidgets.QGridLayout(frame)

        self.epochBox = None
        self.epochLbl = None

        # Set up file type, error flag, and time format layouts
        self.fileTypeLt = self.getFileTypeLayout()
        self.errorFlagLt = self.getErrorFlagLayout()
        self.timeLt = self.getTimeLt()

        layout.addLayout(self.fileTypeLt, 0, 0, 1, 3)
        layout.addLayout(self.timeLt, 1, 0, 1, 3)
        layout.addLayout(self.errorFlagLt, 2, 0, 1, 3)

        # Add in apply button
        self.applyBtn = QtWidgets.QPushButton('Apply')
        layout.addWidget(self.applyBtn, 3, 1, 1, 1)

        # Set up or remove epoch setting layout depending on default time format
        self.timeFormatChanged()

    def getFileTypeLayout(self):
        layout = QtWidgets.QGridLayout()
        self.fileTypeBox = QtWidgets.QComboBox()
        self.fileTypeBox.addItems(['CSV', 'TSV', 'Fixed Columns'])
        self.addPair(layout, 'File Type: ', self.fileTypeBox, 0, 0, 1, 1)
        return layout

    def getErrorFlagLayout(self):
        layout = QtWidgets.QGridLayout()
        # Set up error flag spinbox
        self.errorFlagBox = QtWidgets.QSpinBox()
        self.errorFlagBox.setPrefix('1e+') # TODO: NEEDS TO RECOGNIZE NANS
        self.errorFlagBox.setMinimum(5)
        self.errorFlagBox.setMaximum(60)
        self.errorFlagBox.setValue(30)
        self.errorFlagBox.setFixedWidth(100)
        self.addPair(layout, 'Error Flag: ', self.errorFlagBox, 0, 0, 1, 1)

        # Add in spacer item
        spacer = QtWidgets.QSpacerItem(0, 0, hPolicy=QSizePolicy.MinimumExpanding)
        layout.addItem(spacer, 0, 2, 1, 1)
        return layout

    def getTimeLt(self):
        layout = QtWidgets.QGridLayout()
        # Set up time column format combobox
        self.timeModeBox = QtWidgets.QComboBox()
        self.timeModeBox.addItems(['Seconds', 'UTC Timestamp'])
        self.timeModeBox.currentTextChanged.connect(self.timeFormatChanged)
        self.addPair(layout, 'Time Format: ', self.timeModeBox, 0, 0, 1, 1)
        return layout

    def getEpochLt(self):
        # Set up a spinbox for the user to specify an epoch year
        layout = QtWidgets.QGridLayout()
        self.epochBox = QtWidgets.QSpinBox()
        self.epochBox.setMinimum(1970)
        self.epochBox.setMaximum(2080)
        lblTxt = 'Epoch Year: '
        self.epochLbl = self.addPair(layout, lblTxt, self.epochBox, 0, 0, 1, 1)
        return layout

    def timeFormatChanged(self):
        # Remove epoch year UI elements if times are not in seconds mode
        if self.timeModeBox.currentText() != 'Seconds':
            for item in [self.epochBox, self.epochLbl]:
                if item is None:
                    continue
                self.timeLt.removeWidget(item)
                item.deleteLater()
        else: # Otherwise, add in epoch UI elements
            epochLt = self.getEpochLt()
            self.timeLt.addLayout(epochLt, 1, 0, 1, 2)

class Asc_Importer(QtWidgets.QDialog, Asc_UI):
    def __init__(self, window, filename, parent=None):
        super().__init__(parent)
        self.window = window
        self.fn = None
        self.setFile(filename)
        self.ui = Asc_UI()
        self.ui.setupUI(self.window, self)

        # Guess default file type and time format settings
        fileType, optionIndex = self.guessFileType(filename, self.header)
        self.ui.fileTypeBox.setCurrentIndex(optionIndex)

        timeMode, optionIndex = self.guessTimeFormat(self.firstLine)
        self.ui.timeModeBox.setCurrentIndex(optionIndex)

    def setFile(self, filename):
        # Store filename, file descriptor, header, and lines in file
        if self.fn:
            self.fn.close()
        self.filename = filename
        self.fn = open(filename, 'r')
        self.header = self.fn.readline()
        self.firstLine = self.fn.readline()
        self.lines = [self.firstLine] + self.fn.readlines()

    def getRefYear(self):
        if self.ui.epochBox:
            return self.ui.epochBox.value()
        else:
            return 1970
    
    def getTimeMode(self):
        return self.ui.timeModeBox.currentText()
    
    def getFileFormat(self):
        return self.ui.fileTypeBox.currentText()
    
    def getErrorFlag(self):
        return 10 ** (self.ui.errorFlagBox.value())

    def getMapEpoch(self, refYear):
        if refYear >= 2000:
            return 'J2000'
        else:
            return 'Y1970'
    
    def getBaseTick(self, epoch, refYear):
        # Map the refYear to a UTC timestamp and then map the timestamp
        # to the number of seconds since the given epoch
        refDt = datetime(refYear, 1, 1)
        ts = refDt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        return self.mapUTCtoTick(ts, epoch)

    def guessFileType(self, filename, header):
        # Guess file type by extension and also return index in options list
        if '.csv' in filename:
            return ('CSV', 0)
        elif '.tsv' in filename:
            return ('TSV', 1)
        else:
            return ('Fixed Columns', 2)
    
    def guessTimeFormat(self, firstLine):
        # Attempts to check if times are in timestamp format by performing
        # a regex match against the first non-header line in the file
        ex = '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}'
        if re.match(ex, firstLine) is not None:
            return ('UTC Timestamp', 1)
        else:
            return ('Seconds', 0)

    def guessColumns(self, header, firstLine):
        # TODO: Adjust ending for left-aligned headers
        leftAligned = True
        splitHeader = header.split(' ')
        if splitHeader[0] == '':
            leftAligned = False

        cols = [0]
        colIndex = 0 # Character index in line
        itemIndex = 0 # Item index for items in split string list
        while itemIndex < len(splitHeader):
            if len(splitHeader[itemIndex]) > 0: # Look for non-empty strings
                if not leftAligned:
                    cols.append(colIndex+len(splitHeader[itemIndex])+1)
                else:
                    if colIndex != 0:
                        cols.append(colIndex)
                colIndex += len(splitHeader[itemIndex])
            colIndex += 1
            itemIndex += 1

        return cols

    def splitLine(self, line, fileType, columns=None):
        # Split lines according to specified format or columns
        line = line.strip('\n')
        if fileType == 'CSV':
            newLine = [e.strip(' ') for e in line.split(',') if e != '']
            newLine = [e for e in newLine if e != '']
        elif fileType == 'TSV':
            newLine = [e.strip('\n').strip(' ') for e in line.split('\t')]
            newLine = [e for e in newLine if e != '']
        else:
            newLine = [line[columns[i]:columns[i+1]].strip(' ') for i in range(0, len(columns) - 1)]

        return newLine

    def splitHeader(self, header, fileType, columns=None):
        # Split header by specified columns or separator value
        units = []
        if fileType == 'CSV':
            header = [e.strip('\n').strip(' ') for e in header.split(',')][1:]
        elif fileType == 'TSV':
            header = [e.strip('\n').strip(' ') for e in header.split('\t')][1:]
        else:
            header = header.strip('\n')
            header = [header[columns[i]:columns[i+1]].strip(' ') for i in range(0, len(columns) - 1)]
            header = header[1:]

        # Separate out any specified units in brackets, using an empty string
        # in place of any missing units
        header, units = self.splitHeaderUnits(header)
        return header, units

    def splitHeaderUnits(self, headerList):
        # Try to extract any units from column headers in (unit) or [unit] format
        units = []
        updatedHeader = []
        for colName in headerList:
            # Split string by first bracket/parens
            splitBrackets = colName.split('[')
            splitParens = colName.split('(')

            # Default units and column names
            colUnits = ''
            updtColName = colName

            # Check the right side of each split for a matching bracket/parens
            # and extract the string enclosed in the brackets
            if len(splitBrackets) > 1:
                unitsSplit = splitBrackets[1].split(']')
                if len(unitsSplit) > 1 and unitsSplit[0] != '':
                    colUnits = unitsSplit[0]
                    updtColName = splitBrackets[0].strip(' ')
            if len(splitParens) > 1:
                unitsSplit = splitParens[1].split(')')
                if len(unitsSplit) > 1 and unitsSplit[0] != '':
                    colUnits = unitsSplit[0]
                    updtColName = splitParens[0].strip(' ')
            
            units.append(colUnits)
            updatedHeader.append(updtColName)
        return updatedHeader, units

    def mapUTCtoTick(self, timestmp, epoch):
        # Map UTC timestamp into a time (library) tuple/struct
        splitStr = timestmp.split('.')
        if len(splitStr) > 1:
            dateStr, msStr = splitStr
        else:
            dateStr = splitStr[0]
            msStr = ''
        fmtStr = '%Y-%m-%dT%H:%M:%S'
        ffFmtStr = '%Y %j %b %d %H:%M:%S'
        utcTime = time.strptime(dateStr, fmtStr)
        if msStr != '':
            msStr = '.' + msStr

        # Map the tuple back to a UTC timestamp in the format that
        # FFTime recognizes and then map this timestamp into a time tick
        ffUtcTime = time.strftime(ffFmtStr, utcTime)+msStr
        return FFTIME(ffUtcTime, Epoch=epoch)._tick

    def mapTimes(self, data, timeMode, refYear=None):
        timeLst = [line[0] for line in data]
        if self.getTimeMode() == 'Seconds':
            # Map seconds to floats and add in the number of seconds between
            # the user-specified year and the epoch
            times = np.array(list(map(float, timeLst)))
            epoch = self.getMapEpoch(refYear)
            baseTick = self.getBaseTick(epoch, refYear)
            times = times + baseTick
        else:
            # Get the year of the first record and use it to determine an
            # appropriate epoch
            fmtStr = '%Y-%m-%dT%H:%M:%S.%f'
            refYear = datetime.strptime(timeLst[0], fmtStr).year
            epoch = self.getMapEpoch(refYear)
            # Map timestamps to time ticks relative to the new epoch
            times = [self.mapUTCtoTick(ts, epoch) for ts in timeLst]
            times = np.array(times)
        return times, epoch

    def readInData(self, header, lines):
        # Try to identify columns in data if in 'Fixed Column' format
        fileType = self.getFileFormat()
        cols = None
        if fileType == 'Fixed Columns':
            cols = self.guessColumns(header, lines[0])

        # Split header into column titles and units
        dstrs, units = self.splitHeader(header, fileType, cols)

        # Split strings for each line
        splitLines = [self.splitLine(line, fileType, cols) for line in lines]

        # Map time column into ticks since an epoch
        timeMode = self.getTimeMode()
        refYear = self.getRefYear() if timeMode == 'Seconds' else None
        times, epoch = self.mapTimes(splitLines, timeMode, refYear=refYear)

        # Map all other other columns in the data to floats
        data = [list(map(float, line[1:])) for line in splitLines]
        data = np.array(data).T

        return times, data, dstrs, units, epoch

    def linkApplyBtn(self, openFunc):
        self.ui.applyBtn.clicked.connect(openFunc)

    def getStateInfo(self):
        # Information used to reload a file from state
        fileType = self.getFileFormat()
        epoch = None if self.getTimeMode() != 'Seconds' else self.getRefYear()

        state = {}
        state['FileType'] = fileType
        state['Epoch'] = epoch
        return state

    def loadStateInfo(self, state):
        # Set the file type
        fileType = state['FileType']
        self.ui.fileTypeBox.setCurrentText(fileType)

        # Set the time format and if necessary, the epoch year
        if state['Epoch'] is None:
            self.ui.timeModeBox.setCurrentText('UTC Timestamp')
        else:
            self.ui.timeModeBox.setCurrentText('Seconds')
            self.ui.epochBox.setValue(state['Epoch'])

    def readFile(self):
        # Read in times, data, labels, and units, 
        hdr, lines = self.header, self.lines
        times, data, labels, units, epoch = self.readInData(hdr, lines)

        # Create an FD object from the loaded data and parameters
        ancInfo = {}
        ancInfo['Times'] = times
        ancInfo['Records'] = data.T
        ancInfo['Units'] = units
        ancInfo['Labels'] = labels
        ancInfo['Epoch'] = epoch
        ancInfo['StateInfo'] = self.getStateInfo()
        ancInfo['TimeMode'] = self.getTimeMode()
        fileType = self.getFileFormat()
        fd = TextFD(self.filename, fileType, ancInfo)

        errFlag = self.getErrorFlag()
        return (times, data, (labels, units, epoch, errFlag, fd))
    
class ASC_Output():
    '''
        Helps write out files in ASCII format
    '''
    def __init__(self, filename, records, header, secondsFmt=True, fileFmt='CSV', cols=[], epoch=None):
        self.filename = filename
        self.records = records
        self.header = header
        self.fileFmt = fileFmt
        self.secondsFmt = secondsFmt
        self.cols = cols
        if self.cols is not None:
            self.cols[-1] -= 1
        self.epoch = epoch

    def fmtToUTC(self, tick):
        ''' Maps tick to UTC timestamp w/ T in middle '''
        ts = FFTIME(tick, Epoch=self.epoch).UTC
        year, doy, month, day, times = ts.split(' ')
        month = datetime.strptime(month, '%b').strftime('%m')

        date = '-'.join([year, month, day])
        newTs = f'{date}T{times}'

        return newTs

    def formatSep(self, line, sep=','):
        ''' Formats records separated by commas or tabs '''
        recordFmts = ['{:.5f}']*len(line)
        record = [fmt.format(val) for fmt, val in zip(recordFmts, line)]

        # Replace ticks w/ UTC timestamp
        if not self.secondsFmt:
            record[0] = self.fmtToUTC(line[0])
        else:
            record[0] = '{:.5f}'.format(line[0])

        return sep.join(record)

    def formatCol(self, line):
        ''' Formats records separated into fixed-length columns '''
        # Determine whether to align left or right
        if self.header[0] == ' ':
            alignStr = '>'
        else:
            alignStr = ''

        # Create formatting string based on alignment and column spacing
        record = []
        prevCol = 0
        for col, val in zip(self.cols[1:], line):
            diff = col - prevCol
            recordFmt = f'{{:{alignStr}{diff}.5f}}'
            record.append(recordFmt.format(val))
            prevCol = col

        # Replace ticks w/ UTC timestamp
        if not self.secondsFmt:
            record[0] = self.fmtToUTC(line[0])

        return ''.join(record)

    def write(self):
        '''
            Formats and writes out ASCII file
        '''
        # Open file and write header
        fd = open(self.filename, 'w')
        fd.write(self.header)

        # Format records
        if self.fileFmt == 'CSV':
            records = [self.formatSep(line) for line in self.records]
        elif self.fileFmt == 'TSV':
            records = [self.formatSep(line, '\t') for line in self.records]
        else:
            records = [self.formatCol(line) for line in self.records]

        # Write lines to file and close
        recordTxt = '\n'.join(records)
        fd.write(recordTxt)
        fd.close()