import FF_File
from FF_Time import FFTIME, leapFile
from FF_File import FF_EPOCH, FF_ID, FF_STATUS, _FF_BASE
from FF_DESC import FFDESC

import numpy as np

def createFF(name, times, dta, labels, units, sources, epoch):
    # Initialize new flat file's name, record shape, and epoch
    nCol = len(labels) + 1
    nRows = len(times)
    recl = 4 + nCol * 4
    newFF = FF_ID(name, status=FF_STATUS.WRITE, recl=recl, ncols=nCol,
        epoch=epoch, version=' MarsPy/MagPy v.2019')
    newFF.open()
    newFF.setParameters(NROWS=nRows)

    # Set the flat file's resolution and start/end times
    strtTime = FFTIME(times[0], Epoch=epoch)
    endTime = FFTIME(times[-1], Epoch=epoch)
    resolution = times[1]-times[0]
    res = FFTIME(resolution, Epoch=epoch)
    newFF.setInfo(FIRST_TIME=strtTime.UTC, LAST_TIME=endTime.UTC,
        RESOLUTION=res.UTC[-12:], OWNER='IGPP/UCLA')

    # Update the column headers
    setFFDescriptors(newFF, labels, sources, units)

    # Write data and times to flat file
    writeDataToFF(newFF, dta, times)
    
    # Close file before exiting
    newFF.close()
    return newFF

def setFFDescriptors(ff, labels, sources, units):
    # Writes the header information to the flat file
    header = '  # NAME      UNITS     SOURCE                    TYPE  LOC'
    ncol = len(labels) + 1 # Add 1 for time
    DESC = FFDESC(ncol, header)
    desc = ['TIME', 'SEC', '', 'T', 0]
    DESC.setDesc(1, desc)

    for i in range(2, ncol + 1):
        desc[0] = labels[i-2]
        desc[1] = units[i-2]
        desc[2] = sources[i-2]
        desc[3] = 'R'
        desc[4] = (i-1) * 4 + 4
        DESC.setDesc(i, desc)
    ff.colDesc = DESC

def writeDataToFF(ff, dta, times):
    ncol = len(dta[0])
    dt = [('time', '>d'), ('data', '>f4', (ncol))]
    nrows = len(times)
    bytesToWrite = np.empty(nrows, dtype=dt)
    for i in range(nrows):
        bytesToWrite[i] = times[i], dta[i]
    ff.DID.file.write(bytesToWrite)