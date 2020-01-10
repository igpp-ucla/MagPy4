from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import FF_File
from FF_Time import FFTIME, leapFile
from dataDisplay import UTCQDate
import bisect
from datetime import datetime

class TimeManager(object):
    # Class that handles and initializes general time-related functions / states
    def __init__(self, minTime, maxTime, epoch, tO = None, tE = None):
        self.minTime = minTime
        self.maxTime = maxTime
        self.tO = tO if tO is not None else self.minTime
        self.tE = tE if tE is not None else self.maxTime
        self.epoch = epoch
        self.dayCutoff = 60 * 60 * 24
        self.hrCutoff = 60 * 60 * 1.5
        self.minCutoff = 10 * 60

    # given the corresponding time array for data (times) and the time (t), calculate index into data array
    def calcDataIndexByTime(self, times, t):
        assert(len(times) >= 2)
        if t <= times[0]:
            return 0
        if t >= times[-1]:
            return len(times)
        b = bisect.bisect_left(times, t) # can bin search because times are sorted
        if b < 0:
            return 0
        elif b >= len(times):
            return len(times) - 1
        return b

    def getTimeLabelMode(self, rng=None):
        if rng is None:
            rng = self.getSelectedTimeRange()
        if rng > self.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            return 'DAY'
        elif rng > self.hrCutoff: # if over hour show hh:mm:ss
            return 'HR'
        elif rng > self.minCutoff: # if over 10 seconds show mm:ss
            return 'MIN'
        else: # else show mm:ss.sss
            return 'MS'

    def getSelectedTimeRange(self):
        return abs(self.tE - self.tO)

    def getTimeLabel(self, rng):
        if rng > self.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            return 'HH:MM'
        elif rng > self.hrCutoff: # if hour show hh:mm:ss
            return 'HH:MM'
        elif rng > self.minCutoff: # if over 10 seconds show mm:ss
            return 'MM:SS'
        else: # else show mm:ss.sss
            return 'MM:SS.SSS'

    def getTimeTicksFromTimeEdit(self, timeEdit):
        t0 = FFTIME(UTCQDate.QDateTime2UTC(timeEdit.start.dateTime()), Epoch=self.epoch)._tick
        t1 = FFTIME(UTCQDate.QDateTime2UTC(timeEdit.end.dateTime()), Epoch=self.epoch)._tick
        return (t0, t1) if t0 < t1 else (t0, t1)

    def getTimestampFromTick(self, tick):
        return FFTIME(tick, Epoch=self.epoch).UTC

    def getDateTimeFromTick(self, tick):
        return UTCQDate.UTC2QDateTime(self.getTimestampFromTick(tick))
    
    def getDateTimeObjFromTick(self, tick):
        dt = UTCQDate.UTC2QDateTime(self.getTimestampFromTick(tick))
        date = dt.date()
        timeObj = dt.time()
        dt = datetime(date.year(), date.month(), date.day(), timeObj.hour(), timeObj.minute(), timeObj.second(), timeObj.msec())
        return dt
    
    def getTickFromTimestamp(self, ts):
        return FFTIME(ts, Epoch=self.epoch)._tick