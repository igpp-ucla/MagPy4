from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from fflib import ff_time
from dateutil import parser
from .dataDisplay import UTCQDate
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
        start = timeEdit.start.dateTime().toPyDateTime()
        end = timeEdit.end.dateTime().toPyDateTime()
        t0 = ff_time.date_to_tick(start, self.epoch)
        t1 = ff_time.date_to_tick(end, self.epoch)
        return (t0, t1) if t0 < t1 else (t0, t1)

    def getTimestampFromTick(self, tick):
        return ff_time.tick_to_ts(tick, self.epoch)

    def getDateTimeFromTick(self, tick):
        return ff_time.tick_to_date(tick, self.epoch)
    
    def datetime_from_tick(self, tick):
        ts = self.getTimestampFromTick(tick)
        fmt = '%Y %j %b %d %H:%M:%S.%f'
        date = datetime.strptime(ts, fmt)
        return date

    def getTickFromDateTime(self, dt):
        return ff_time.date_to_tick(dt, self.epoch)

    def getDateTimeObjFromTick(self, tick):
        return ff_time.tick_to_date(tick, self.epoch)
    
    def getTickFromTimestamp(self, ts):
        date = parser.isoparse(ts)
        tick = ff_time.date_to_tick(date, self.epoch)
        return tick
    
    def setTimeEditByTicks(self, t0, t1, timeEdit):
        dt0 = self.getDateTimeFromTick(t0)
        dt1 = self.getDateTimeFromTick(t1)
        timeEdit.start.setDateTime(dt0)
        timeEdit.end.setDateTime(dt1)