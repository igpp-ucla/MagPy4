
from PyQt5 import QtGui, QtCore, QtWidgets
from dataDisplay import UTCQDate
import pyqtgraph as pg
import pyqtgraph.exporters
from FF_Time import FFTIME
from datetime import datetime, timedelta
from timeManager import TimeManager
from math import ceil
import functools

# Label item placed at top
class RegionLabel(pg.InfLineLabel):
    def __init__(self, lines, text="", movable=False, position=0.5, anchors=None, **kwds):
        self.lines = lines
        pg.InfLineLabel.__init__(self, lines[0], text=text, movable=False, position=position, anchors=None, **kwds)
        self.lines[1].sigPositionChanged.connect(self.valueChanged)
        self.setAnchor((0.5, 0))

    def updatePosition(self):
        # update text position to relative view location along line
        self._endpoints = (None, None)
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return
        pt = pt2 * self.orthoPos + pt1 * (1-self.orthoPos)
        # Calculate center point between region boundaries
        x1, x2 = self.lines[0].x(), self.lines[1].x()
        diff = abs(x1-x2)/2
        if x1 > x2: # Switch offset if first line crossed second line
            diff *= -1
        self.setPos(QtCore.QPointF(pt.x(), -diff))

class LinkedRegion(pg.LinearRegionItem):
    '''
        Generates and manages a group of linear sub regions for a set of plots
    '''

    # sigRegionActivated is emitted when the region is changed or clicked on
    sigRegionActivated = QtCore.pyqtSignal(object)

    def __init__(self, window, plotItems, values=(0, 1), mode=None, color=None,
        updateFunc=None, linkedTE=None, lblPos='top'):
        self.window = window
        self.plotItems = plotItems
        self.regionItems = []
        self.labelText = mode if mode is not None else ''
        self.fixedLine = False
        self.updateFunc = updateFunc
        self.linkedTE = linkedTE
        self.lblPos = lblPos

        pg.LinearRegionItem.__init__(self, values=(0,0))

        for plt in self.plotItems:
            # Create a LinearRegionItem for each plot with same initialized vals
            regionItem = LinkedSubRegion(self, values=values, color=color)
            regionItem.setBounds([self.window.minTime - self.window.tickOffset, self.window.maxTime-self.window.tickOffset])
            plt.addItem(regionItem)
            self.regionItems.append(regionItem)
            # Connect plot's regions/lines to each other
            line0, line1 = regionItem.lines
            line0.sigDragged.connect(functools.partial(self.linesChanged, line0, 0))
            line1.sigDragged.connect(functools.partial(self.linesChanged, line1, 1))

        # Initialize region label at top-most plot
        self.labelPltIndex = 0 if lblPos == 'top' else len(self.plotItems) - 1
        self.setLabel(self.labelPltIndex)
        if self.linkedTE:
            self.updateTimeEditByLines(self.linkedTE)

    def setLinkedTE(self, te):
        self.linkedTE = te

    def setLabelText(self, lbl):
        if len(self.regionItems) < 1:
            return
        self.regionItems[self.labelPltIndex].lines[0].label.setFormat(lbl)
        self.labelText = lbl

    def setLabel(self, plotNum, pos=0.95):
        if len(self.regionItems) < 1:
            self.labelPlotIndex = 0
            return

        # Create new label for line
        if self.lblPos == 'bottom':
            pos = 0.25
        fillColor = pg.mkColor('#212121') # Dark grey background color
        fillColor.setAlpha(200)
        opts = {'movable': False, 'position':pos, 'color': pg.mkColor('#FFFFFF'),
            'fill': fillColor}
        label = RegionLabel(self.regionItems[plotNum].lines, text=self.labelText, **opts)
        # Update line's label and store the plot index where it's currently located
        self.regionItems[plotNum].lines[0].label = label
        self.labelPltIndex = plotNum

    def mouseDragEvent(self, ev):
        # Called by sub-regions to drag all regions at same time
        for regIt in self.regionItems:
            pg.LinearRegionItem.mouseDragEvent(regIt, ev)

        # Activate current region if starting a mouse drag
        if ev.isStart():
            self.onRegionActivated()

        # Signal that region has changed
        self.onRegionChanged()

    def linesChanged(self, line, lineNum, extra):
        # Update all lines on same side of region as the original 'sub-line'
        pos = line.value()
        for regItem in self.regionItems:
            lines = regItem.lines
            lines[lineNum].setValue(pos)
            if self.isFixedLine():
                lines[int(not lineNum)].setValue(pos)
            regItem.prepareGeometryChange()
        self.onRegionChanged()

    def onRegionActivated(self):
        # Update time edit and emit signal
        self.updateTimeEditByLines(self.linkedTE)
        self.sigRegionActivated.emit(self)

    def onRegionChanged(self):
        # Update time edit and call update function
        self.updateTimeEditByLines(self.linkedTE)
        if self.updateFunc:
            self.updateFunc()

    def setRegion(self, rgn):
        for regIt in self.regionItems:
            regIt.setRegion(rgn)

    def getRegion(self):
        # Return region's ticks (with offset added back in)
        if self.regionItems == []:
            return [0,0]
        else:
            x1, x2 = self.regionItems[0].getRegion() # Use any region's bounds
            return (x1+self.window.tickOffset, x2+self.window.tickOffset)

    def removeRegionItems(self):
        # Remove all sub-regions from corresponding plots
        for plt, regIt in zip(self.plotItems, self.regionItems):
            plt.removeItem(regIt)

    def linePos(self, lineNum=0):
        # Returns the line position for this region's left or right bounding line
        if self.regionItems == []:
            return 0
        return self.regionItems[0].lines[lineNum].value()

    def setFixedLine(self, val=True):
        self.fixedLine = val
    
    def isFixedLine(self):
        return self.fixedLine

    def isLine(self):
        # Returns True if region lines are at same position
        return (self.linePos(0) == self.linePos(1))

    def isVisible(self, plotIndex):
        # Checks if sub-region at given plot num is set to visible
        return self.regionItems[plotIndex].isVisible()

    def setVisible(self, visible, plotIndex):
        # Updates visibility of the sub-region at the given plot num
        self.regionItems[plotIndex].setVisible(visible)
        label = self.regionItems[self.labelPltIndex].lines[0].label
        # If removing sub-region with label
        if self.labelPltIndex == plotIndex and not visible:
            label.setVisible(False)
            if self.lblPos == 'top':
                plotIndex += 1
                # Find the next visible sub-region and place the label there
                while plotIndex < len(self.regionItems):
                    if self.regionItems[plotIndex].isVisible():
                        self.setLabel(plotIndex)
                        return
                    plotIndex += 1
            else: # Look for last visible sub-region
                plotIndex -= 1
                while plotIndex > 0:
                    if self.regionItems[plotIndex].isVisible():
                        self.setLabel(plotIndex)
                        return
                    plotIndex -= 1
        # If adding in a sub-region at plot index higher than the one
        # with the region label, move the label to this plot
        elif self.lblPos == 'top' and plotIndex < self.labelPltIndex and visible:
            label.setVisible(False)
            self.setLabel(plotIndex)
        # In case of linked region w/ label at bottom, set label on
        # the lowest visible sub-region
        elif self.lblPos == 'bottom' and plotIndex > self.labelPltIndex and visible:
            label.setVisible(False)
            self.setLabel(plotIndex)

    def setAllRegionsVisible(self, val=True):
        for subRegion in self.regionItems:
            subRegion.setVisible(val)

    def updateTimeEditByLines(self, timeEdit):
        if timeEdit is None:
            return

        x0, x1 = self.getRegion()
        t0 = UTCQDate.UTC2QDateTime(FFTIME(x0, Epoch=self.window.epoch).UTC)
        t1 = UTCQDate.UTC2QDateTime(FFTIME(x1, Epoch=self.window.epoch).UTC)

        timeEdit.setStartNoCallback(min(t0,t1))
        timeEdit.setEndNoCallback(max(t0,t1))

    def setMovable(self, val):
        for item in self.regionItems:
            item.setMovable(val)

    def mouseClickEvent(self, ev):
        # Signal this region to be set as the active region
        self.onRegionActivated()

class LinkedSubRegion(pg.LinearRegionItem):
    def __init__(self, grp, values=(0, 1), color=None, orientation='vertical', brush=None, pen=None):
        self.grp = grp

        pg.LinearRegionItem.__init__(self, values=values)

        # Set region fill color to chosen color with some transparency
        if color is not None:
            color = pg.mkColor(color)
            linePen = pg.mkPen(color)
            color.setAlpha(20)
        else: # Default fill and line colors
            color = pg.mkColor('#b3b7f9')
            linePen = pg.mkPen('#000cff')
            color.setAlpha(35)
        self.setBrush(pg.mkBrush(color))

        # Set line's pen to same color but opaque and its hover pen to black
        for line in self.lines:
            pen = pg.mkPen('#000000')
            pen.setWidth(2)
            line.setHoverPen(pen)
            line.setPen(linePen)

    def mouseClickEvent(self, ev):
        pg.LinearRegionItem.mouseClickEvent(self, ev)
        self.grp.mouseClickEvent(ev)

    def mouseDragEvent(self, ev):
        # If this sub-region is dragged, move all other sub-regions accordingly
        self.grp.mouseDragEvent(ev)
    
    def mouseClickEvent(self, ev):
        self.grp.mouseClickEvent(ev)
        pg.LinearRegionItem.mouseClickEvent(self, ev)


class MagPyPlotItem(pg.PlotItem):
    def isSpecialPlot(self):
        return False

    def updateLogMode(self):
        x = self.ctrl.logXCheck.isChecked()
        y = self.ctrl.logYCheck.isChecked()
        for i in self.items:
            if hasattr(i, 'setLogMode'):
                i.setLogMode(x,y)
        self.getAxis('bottom').setLogMode(x)
        self.getAxis('top').setLogMode(x)
        self.getAxis('left').setLogMode(y)
        self.getAxis('right').setLogMode(y)
    
class MagPyColorPlot(MagPyPlotItem):
    def __init__(self, *args, **kwargs):
        MagPyPlotItem.__init__(self, *args, **kwargs)
        self.actionLink = None

    def isSpecialPlot(self):
        return True

    def getContextMenus(self, event):
        if self.actionLink:
            self.stateGroup.autoAdd(self.actionLink)
            return [self.ctrlMenu, self.actionLink]
        else:
            return self.ctrlMenu

class MagPyAxisItem(pg.AxisItem):
    def __init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True):
        self.tickDiff = None
        pg.AxisItem.__init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True)

    def setCstmTickSpacing(self, diff):
        self.setTickSpacing(major=diff, minor=diff*1000)
        self.tickDiff = diff

    def resetTickSpacing(self):
        self._tickSpacing = None
        self.picture = None
        self.update()

    def axisType(self):
        return 'Regular' if not self.logMode else 'Log'

    def calcDesiredWidth(self):
        w = 0
        if not self.style['showValues']:
            w = 0
        elif self.style['autoExpandTextSpace'] is True:
            w = self.textWidth
        else:
            w = self.style['tickTextWidth']
        w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
        w += max(0, self.style['tickLength'])
        if self.label.isVisible():
            w += self.label.boundingRect().height() * 0.8  # bounding rect is usually an overestimate
        return w + 10 # 10 extra to offset if in scientific notation

    def tickSpacing(self, minVal, maxVal, size):
        if self.logMode:
            return LogAxis.tickSpacing(self, minVal, maxVal, size)
        else:
            return pg.AxisItem.tickSpacing(self, minVal, maxVal, size)

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        self.tickDiff = None
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont = QtGui.QFont()
        self.tickFont.setPixelSize(14)

    def setLogMode(self, log):
        if log:
            self.style['maxTextLevel'] = 1 # never have any subtick labels
            self.style['textFillLimits'] = [(0,1.1)] # try to always draw labels
        else:
            self.style['maxTextLevel'] = 2
            self.style['textFillLimits'] = [(2, 0.7)] # try to always draw labels

        if log != self.logMode:
            self.tickDiff = None

        pg.AxisItem.setLogMode(self, log)

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below
        return pg.AxisItem.tickStrings(self,values,scale,spacing)

    def tickSpacing(self, minVal, maxVal, size):
        if self.logMode:
            if self.tickDiff:
                return [(self.tickDiff, 0), (self.tickDiff, 0)]
            else:
                return [(10.0,0),(1.0,0),(0.5,0)]
        return pg.AxisItem.tickSpacing(self,minVal,maxVal,size)

    def axisType(self):
        if self.logMode:
            return 'Log'
        else:
            return 'Regular'

    # overriden from source to be able to have superscript text
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        if not self.logMode:
            pg.AxisItem.drawPicture(self, p, axisSpec, tickSpecs, textSpecs)
            return
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        
        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity
        
        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        ## Draw all text
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            qst = QtGui.QStaticText(f'10<sup>{text}</sup>')
            qst.setTextFormat(QtCore.Qt.RichText)
            p.drawStaticText(rect.left(), rect.top(), qst)
            #p.drawText(rect, flags, text)
            #p.drawRect(rect)

    def setCstmTickSpacing(self, diff):
        self.setTickSpacing(major=diff, minor=diff)
        self.tickDiff = diff

    def resetTickSpacing(self):
        self._tickSpacing = None
        self.tickDiff = None
        self.picture = None
        self.update()

# subclass based off example here:
# https://github.com/ibressler/pyqtgraph/blob/master/examples/customPlot.py
class DateAxis(pg.AxisItem):
    def __init__(self, epoch, orientation, offset=0, pen=None, linkView=None, parent=None,
                maxTickLength=-5, showValues=True):
        self.tickOffset = offset
        self.timeRange = None
        self.epoch = epoch
        self.tm = TimeManager(0, 0, self.epoch)
        pg.AxisItem.__init__(self, orientation, pen, linkView, None,
                            maxTickLength,showValues)

        # Dictionary holding default increment values for ticks
        self.modeToDelta = {}
        self.modeToDelta['DAY'] = timedelta(hours=6)
        self.modeToDelta['HR'] = timedelta(minutes=30)
        self.modeToDelta['MIN'] = timedelta(minutes=5)
        self.modeToDelta['MS'] = timedelta(milliseconds=500)

        # Increment values in seconds respective to each time unit
        # Year, month, day, hours, minutes, seconds, ms
        self.addVals = [24*60*60, 24*60*60, 24*60*60, 60*60, 60, 1, 0]

        # Custom interval base lists to use for hours, min, sec, ms; See tickSpacing()
        refDays = np.array([1,3, 6, 12]) * self.addVals[-5]
        refHrs = np.array([1, 2, 6, 12]) * self.addVals[-4]
        refMin = np.array([1, 2, 5, 10, 15, 30]) * self.addVals[-3]
        refSec = np.array([1, 2, 5, 10, 15, 30]) * self.addVals[-2]
        refMs = np.array([0.1, 0.25, 0.5])

        self.intervalLists = [None, None, refDays, np.concatenate([refHrs, refDays]), 
            np.concatenate([refMin, refHrs, refDays]), np.concatenate([refSec, refMin, refHrs]), 
            np.concatenate([refMs, refSec, refMin, refHrs])]

        # String used by strftime/strptime to parse UTC strings
        self.fmtStr = '%Y %j %b %d %H:%M:%S.%f'

        # Custom tick spacing
        self.tickDiff = None

        # Custom label formatting
        self.labelFormat = None
        self.timeModes = ['DATE', 'DATE HH:MM', 'DATE HH:MM:SS', 'HH',
            'HH:MM', 'HH:MM:SS', 'MM', 'MM:SS', 'MM:SS.SSS', 'SS.SSS']
        self.refIndices = [[1,2], [1,2,3,4], [1,2,3,4,5], [3], [3,4],
            [3,4,5], [4], [4,5], [4,5,6], [5,6]] # Sections of time to use for each time mode
        self.kwSeparators = [' ', ' ', ' ', ':', ':', '.', ' '] # Suffix for each time section
        self.timeDict = {} # Maps time modes to indices
        for timeMode, indexLst in zip(self.timeModes, self.refIndices):
            self.timeDict[timeMode] = indexLst

    def splitTimestamp(self, tmstmp):
        # Break down timestamp into its components and return a list of strings
        dateSplit = tmstmp.split(' ')
        year = dateSplit[0]
        month = dateSplit[2]
        day = dateSplit[3]

        # Split time into its units
        timeSplit1 = dateSplit[4].split(':')
        timeSplit2 = timeSplit1[2].split('.')
        hours, minutes = timeSplit1[0:2]
        seconds, ms = timeSplit2[0:2]

        return [year, month, day, hours, minutes, seconds, ms]

    def setLabelFormat(self, kw):
        self.labelFormat = kw
        # Update axis label
        if self.label and (self.label.isVisible()):
            kw = kw.strip('DATE ')
            if kw == '':
                kw = 'Time'
            self.setLabel(kw)
        self.picture = None
        self.update()

    def resetLabelFormat(self):
        self.labelFormat = None
        # Reset axis label
        if self.label and (self.label.isVisible()):
            label = self.tm.getTimeLabel(self.tm.getSelectedTimeRange())
            self.setLabel(label)
        self.picture = None
        self.update()

    def setRange(self, mn, mx):
        pg.AxisItem.setRange(self, mn, mx)
        self.tm.tO = mn + self.tickOffset
        self.tm.tE = mx + self.tickOffset

    # Format a timestamp according to format underneath axis
    def fmtTimeStmp(self, times):
        # Get a formatted timestamp based on user-set format or default format
        splits = times.split(' ')
        t = splits[4]

        if self.labelFormat:
            fmt = self.labelFormat
            return self.specificSplitStr(times, fmt)
        else:
            fmt = self.defaultLabelFormat()
            return self.specificSplitStr(times, fmt)

    def getLabelFormat(self):
        if self.labelFormat:
            return self.labelFormat
        else:
            return self.defaultLabelFormat()

    def specificSplitStr(self, timestamp, fmt):
        # Get indices associated with label format and split timestamp into components
        indices = self.timeDict[fmt]
        splitStr = self.splitTimestamp(timestamp)

        # Build timestamp from components corresp. to indices, using the
        # appropriate suffix/prefix as needed
        newStr = ''
        for index in indices[::-1]:
            if newStr != '':
                sep = self.kwSeparators[index]
                newStr = splitStr[index] + sep + newStr
            else:
                newStr = splitStr[index]
        return newStr

    def defaultLabelFormat(self):
        # Get time label format based on range visible
        if self.timeRange is not None:
            rng = abs(self.timeRange[1] - self.timeRange[0])
        else:
            rng = self.tm.getSelectedTimeRange()
        
        fmt = 'MS' # Show mm:ss.mmm

        if rng > self.tm.dayCutoff: # if over day show MMM dd hh:mm:ss
            fmt = 'DAY'
        elif rng > self.tm.hrCutoff: # if over half hour show hh:mm:ss
            fmt = 'HR'
        elif rng > self.tm.minCutoff: # if over 10 seconds show mm:ss
            fmt = 'MIN'
        
        fmt = self.formatSplitStr(fmt)
        return fmt

    def getDefaultLabel(self):
        return self.defaultLabelFormat().strip('DATE ')

    def formatSplitStr(self, fmt='MS'):
        # Format timestamp based on selected keyword
        if fmt == 'DAY':
            return 'DATE HH:MM'
        elif fmt == 'HR':
            return 'HH:MM'
        elif fmt == 'MIN':
            return 'MM:SS'
        else:
            return 'MM:SS.SSS'

    # Helper functions for convert between datetime obj, timestamp, and tick val
    def tmstmpToDateTime(self, timeStr):
        return datetime.strptime(timeStr, self.fmtStr)
    def dateTimeToTmstmp(self, dt):
        return dt.strftime(self.fmtStr)[:-3] # Remove extra 3 millisecond digits
    def tickStrToVal(self, tickStr):
        return (FFTIME(tickStr, Epoch=self.window.epoch)).tick

    # Used to zero out insignificant values in a datetime object (wrt time label res)
    def zeroLowerVals(self, dt, delta):
        if delta >= timedelta(days=1):
            dt = dt.replace(day=0)
        if delta >= timedelta(hours=1):
           dt = dt.replace(hour = 0)
        if delta >= timedelta(minutes=1):
            dt = dt.replace(minute = 0)
        if delta >= timedelta(seconds=1):
            dt = dt.replace(second = 0)
        if delta >= timedelta(milliseconds=1):
            dt = dt.replace(microsecond = 0)
        return dt

    def tickSpacing(self, minVal, maxVal, size):
        if self.tickDiff:
            return [(self.tickDiff.total_seconds(), 0)]
        else:
            spacing = self.defaultTickSpacing(minVal, maxVal, size)
            return spacing

    def timeSpacingBase(self):
        fmt = self.getLabelFormat()
        indexLst = self.timeDict[fmt]
        minIndex = max(indexLst[-1], 2) # Only need to offset times
        return minIndex, self.addVals[minIndex]

    def defaultTickSpacing(self, minVal, maxVal, size):
        # First check for override tick spacing
        if self._tickSpacing is not None:
            return self._tickSpacing

        dif = abs(maxVal - minVal)
        if dif == 0:
            return []
        
        ## decide optimal minor tick spacing in pixels (this is just aesthetics)
        optimalTickCount = max(2., np.log(size))
        
        ## optimal minor tick spacing 
        optimalSpacing = dif / optimalTickCount
        
        intervals = np.array([1., 2., 10., 20., 100.])
        minIndex, base = self.timeSpacingBase()
        refIntervals = self.intervalLists[minIndex]
        if refIntervals is not None:
            intervals = refIntervals
        elif base > 0:
            intervals = intervals * base
        else:
            p10unit = 10 ** np.floor(np.log10(optimalSpacing))
            intervals = intervals * p10unit  

        ## Determine major/minor tick spacings which flank the optimal spacing.
        minorIndex = 0
        while (minorIndex + 2 < len(intervals)) and intervals[minorIndex+1] < optimalSpacing:
            minorIndex += 1

        upperIndex = minorIndex + 1
        lowerIndex = minorIndex 

        levels = [
            (intervals[upperIndex], 0),
            (intervals[lowerIndex], 0),
        ]
        
        if refIntervals is not None:
            level = intervals[minorIndex]
            levels = [
                (level*2, 0),
                (level, 0)
            ]
            return levels

        if self.style['maxTickLevel'] >= 2:
            ## decide whether to include the last level of ticks
            minSpacing = min(size / 20., 30.)
            maxTickCount = size / minSpacing
            if dif / intervals[minorIndex] <= maxTickCount:
                levels.append((intervals[minorIndex], 0))
            return levels

    def getZeroPaddedInt(self, val):
        if val < 10:
            return '0'+str(int(val))
        else:
            return str(int(val))

    def getOffset(self, minVal, maxVal):
        fmt = self.getLabelFormat()
        indexLst = self.timeDict[fmt]
        spacers = [' ', ' ', ' ', ':', ':', '.']
        zeroVals = ['', '', '', '00', '00', '00', '000'] # Hours, minutes, seconds, ms

        # Convert minVal to a timestamp and then to its elements
        baseTick = minVal
        baseUTC = (FFTIME(baseTick, Epoch=self.tm.epoch)).UTC
        splitUTC = self.splitTimestamp(baseUTC)
        splitUTC[1] = datetime.strptime(splitUTC[1], '%b').strftime('%m')

        # Extract day of year and map month to a number
        doy = baseUTC.split(' ')[1]

        # Zero out lower values in the split timestamp
        # year, month, day, hours, minutes, seconds, ms
        minIndex = max(indexLst[-1], 2) # Only need to offset times
        count = 0
        for z in range(minIndex+1, 7):
            if splitUTC[z] != zeroVals[z]:
                count += 1
            splitUTC[z] = zeroVals[z]

        # Build timestamp from edited values
        splitUTC[1] = datetime.strptime(splitUTC[1], '%m').strftime('%b')
        newUTC = splitUTC[0] + ' ' + doy
        for z in range(1, 7):
            newUTC = newUTC + spacers[z-1] + splitUTC[z]

        newTick = (FFTIME(newUTC, Epoch=self.tm.epoch)).tick

        # If zero-ed out timestamp is different from original timestamp
        # increment the smallest time unit that will be visible in the
        # tick label
        if minIndex > 2 and minIndex < 6 and count > 0:
            newTick = newTick + self.addVals[minIndex]
        
        # Return the difference between the original and new starting ticks
        # if the difference is positive
        diff = newTick - baseTick

        if diff < 0:
            return 0
        else:
            return diff

    def tickValues(self, minVal, maxVal, size):
        # Get tick values for standard range and then subtract the offset
        # (keeps the spacing on neat numbers (i.e. 25 vs 23.435 seconds)
        # since offset must be added back in when generating strings)
        minVal = minVal + self.tickOffset
        maxVal = maxVal + self.tickOffset
        vals = pg.AxisItem.tickValues(self, minVal, maxVal, size)
        newVals = []
        for ts, tlst in vals:
            if len(tlst) < 1:
                continue
            # Adjust starting tick for each spacing group to start
            # on a neat number
            diff = self.getOffset(min(tlst), max(tlst))
            newtlst = [v - self.tickOffset + diff for v in tlst]
            newVals.append((ts, newtlst))
        return newVals

    def tickStrings(self, values, scale, spacing):
        # Convert start/end times to strings
        strings = []
        for v in values:
            ts = (FFTIME(v+self.tickOffset, Epoch=self.tm.epoch)).UTC
            s = self.fmtTimeStmp(ts)
            strings.append(s)
        return strings

    def setCstmTickSpacing(self, diff):
        self.tickDiff = diff
        self.picture = None
        self.update()

    def resetTickSpacing(self):
        self.tickDiff = None
        self.picture = None
        self.update()

    def axisType(self):
        return 'DateTime'

class GridGraphicsLayout(pg.GraphicsLayout):
    def __init__(self, window=None, *args, **kwargs):
        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.window = window
        self.lastWidth = 0
        self.lastHeight = 0

    def clear(self):  # clear doesnt get rid of grid layout formatting correctly, todo: make override of this
        pg.GraphicsLayout.clear(self)
        self.currentRow = 0
        self.currentCol = 0

    def paint(self, p, *args):
        pg.GraphicsLayout.paint(self,p,*args)
        if not self.window:
            return
        vr = self.viewRect()
        w = vr.width()
        h = vr.height()
        if w != self.lastWidth or h != self.lastHeight:
            self.lastWidth = w
            self.lastHeight = h

class SpectraPlotItem(MagPyPlotItem):
    # plotItem subclass so we can set Spectra plots to be square
    def __init__(self, window=None, *args, **kargs):
        self.squarePlot = False
        pg.PlotItem.__init__(self, *args, **kargs)

    def commonResize(self, w, h):
        if not self.squarePlot:
            pg.PlotItem.resize(self, w, h)
            return
        # Set plot heights and widths to be the same (accounting for difference in
        # height due to title), so left/bottom axes are same length
        titleht = 15
        if h > w:
            pg.PlotItem.resize(self, w, w + titleht)
        elif h < w:
            pg.PlotItem.resize(self, h + titleht, h)

    def resize(self, w, h):
        self.commonResize(w, h)

    def resize(self, sze):
        h, w = sze.height(), sze.width()
        self.commonResize(w, h)

    def resizeEvent(self, event):
        if self.squarePlot:
            sze = self.boundingRect().size()
            self.resize(sze)
        else:
            pg.PlotItem.resizeEvent(self, event)

class StackedAxisLabel(pg.GraphicsLayout):
    def __init__(self, lbls, angle=90, *args, **kwargs):
        self.lblTxt = lbls
        self.sublabels = []
        self.angle = angle
        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.layout.setVerticalSpacing(-2)
        self.layout.setHorizontalSpacing(-2)
        if angle != 0:
            self.layout.setContentsMargins(5, 0, 5, 0)
        else:
            self.layout.setContentsMargins(0, 0, 0, 0)
        self.setupLabels(lbls)

    def getLabelText(self):
        return self.lblTxt

    def setupLabels(self, lbls):
        if self.angle > 0:
            lbls = lbls[::-1]
        if self.angle == 0 or self.angle == -180:
            self.layout.setRowStretchFactor(0, 1)
        for i in range(0, len(lbls)):
            lbl = lbls[i]
            sublbl = pg.LabelItem(lbl, angle=self.angle)
            if self.angle == 0 or self.angle == -180:
                self.addItem(sublbl, i+1, 0, 1, 1)
            else:
                self.addItem(sublbl, 0, i, 1, 1)
            self.sublabels.append(sublbl)
        if self.angle == 0 or self.angle == -180:
            self.layout.setRowStretchFactor(len(lbls)+1, 1)

    def adjustLabelSizes(self, plotHeight):
        if self.sublabels == []:
            return

        # Get the max length of each line of text in this label
        maxChar = max(list(map(len, self.lblTxt)))

        # Increment the font size until the average char width * max line length
        # is greater than the plot height
        font = QtGui.QFont()
        currSize = 4
        font.setPointSize(currSize)
        met = QtGui.QFontMetricsF(font)

        while met.averageCharWidth() * maxChar < plotHeight:
            currSize += 1
            font.setPointSize(currSize)
            met = QtGui.QFontMetricsF(font)

        # Set the label font size to the calculated point size - 1
        fontSize = min(max(2, font.pointSize() - 1), 12)
        for lbl in self.sublabels:
            txt = lbl.text
            lbl.setText(txt, size=str(fontSize)+'pt')

class BLabelItem(pg.LabelItem):
    def setHtml(self, html):
        self.item.setHtml(html)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

class RowGridLayout(pg.GraphicsLayout):
    def __init__(self, maxCols=None, parent=None, border=None):
        self.maxCols = maxCols
        pg.GraphicsLayout.__init__(self)

    def clear(self):
        pg.GraphicsLayout.clear(self)
        self.currentRow = 0
        self.currentCol = 0

    def setNumCols(self, n):
        self.maxCols = n

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        pg.GraphicsLayout.addItem(self, item, row, col, rowspan, colspan)

        # If current column (after item is placed) is >= maxCols,
        # move to the next row
        if self.maxCols is not None and self.currentCol >= self.maxCols:
            self.nextRow()

    def getRow(self, rowNum):
        # Get list of column numbers in sorted order and return items in row
        cols = list(self.rows[rowNum].keys())
        cols.sort()
        return [self.rows[rowNum][col] for col in cols]

    def getRowItems(self):
        # Get list of row numbers in sorted order
        rowIndices = list(self.rows.keys())
        rowIndices.sort()

        # Get list of items in each row
        rowItems = []
        for row in rowIndices:
            rowItems.append(self.getRow(row))

        return rowItems

# same as pdi but with better down sampling (bds)
class PlotDataItemBDS(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        pg.PlotDataItem.__init__(self, *args, **kwargs)

class MagPyPlotDataItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        pg.PlotDataItem.__init__(self, *args, **kwargs)

class LinkedAxis(MagPyAxisItem):
    def calcDesiredWidth(self):
        w = 0
        if not self.style['showValues']:
            w = 0
        elif self.style['autoExpandTextSpace'] is True:
            w = self.textWidth
        else:
            w = self.style['tickTextWidth']
        w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
        w += max(0, self.style['tickLength'])
        if self.label.isVisible():
            w += self.label.boundingRect().height() * 0.8  # bounding rect is usually an overestimate
        return w + 10 # 10 extra to offset if in scientific notation

# Same as export() from pyqtgraph's ImageExporter class
# But fixed bug where width/height params are not set as integers
def cstmImageExport(self, fileName=None, toBytes=False, copy=False):
    if fileName is None and not toBytes and not copy:
        filter = ["*."+bytes(f).decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
        preferred = ['*.png', '*.tif', '*.jpg']
        for p in preferred[::-1]:
            if p in filter:
                filter.remove(p)
                filter.insert(0, p)
        self.fileSaveDialog(filter=filter)
        return
        
    targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
    sourceRect = self.getSourceRect()
    
    w, h = self.params['width'], self.params['height']
    if w == 0 or h == 0:
        raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w,h))
    bg = np.empty((int(self.params['width']), int(self.params['height']), 4), dtype=np.ubyte)
    color = self.params['background']
    bg[:,:,0] = color.blue()
    bg[:,:,1] = color.green()
    bg[:,:,2] = color.red()
    bg[:,:,3] = color.alpha()
    self.png = pg.makeQImage(bg, alpha=True)
    
    ## set resolution of image:
    origTargetRect = self.getTargetRect()
    resolutionScale = targetRect.width() / origTargetRect.width()

    painter = QtGui.QPainter(self.png)

    try:
        self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'], 'painter': painter, 'resolutionScale': resolutionScale})
        painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
        self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
    finally:
        self.setExportMode(False)
    painter.end()
    
    if copy:
        QtGui.QApplication.clipboard().setImage(self.png)
    elif toBytes:
        return self.png
    else:
        self.png.save(fileName)

pg.exporters.ImageExporter.export = cstmImageExport

import xml.dom.minidom as xml
import numpy as np
import re
from pyqtgraph.python2_3 import asUnicode
from pyqtgraph.Qt import QtSvg, QT_LIB

xmlHeader = """\
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"  version="1.2" baseProfile="tiny">
<title>pyqtgraph SVG export</title>
<desc>Generated with Qt and pyqtgraph</desc>
"""

def cstmSVGExport(self, fileName=None, toBytes=False, copy=False):
    if toBytes is False and copy is False and fileName is None:
        self.fileSaveDialog(filter="Scalable Vector Graphics (*.svg)")
        return

    ## Qt's SVG generator is not complete. (notably, it lacks clipping)
    ## Instead, we will use Qt to generate SVG for each item independently,
    ## then manually reconstruct the entire document.
    options = {ch.name():ch.value() for ch in self.params.children()}
    xml = generateSvg(self.item, options)

    if toBytes:
        return xml.encode('UTF-8')
    elif copy:
        md = QtCore.QMimeData()
        md.setData('image/svg+xml', QtCore.QByteArray(xml.encode('UTF-8')))
        QtGui.QApplication.clipboard().setMimeData(md)
    else:
        with open(fileName, 'wb') as fh:
            fh.write(asUnicode(xml).encode('utf-8'))

def generateSvg(item, options={}):
    global xmlHeader
    try:
        node, defs = cstmGenerateItemSvg(item, options=options)
    finally:
        ## reset export mode for all items in the tree
        if isinstance(item, QtGui.QGraphicsScene):
            items = item.items()
        else:
            items = [item]
            for i in items:
                items.extend(i.childItems())
        for i in items:
            if hasattr(i, 'setExportMode'):
                i.setExportMode(False)

    cleanXml(node)
    defsXml = "<defs>\n"
    for d in defs:
        defsXml += d.toprettyxml(indent='    ')
    defsXml += "</defs>\n"
    return xmlHeader + defsXml + node.toprettyxml(indent='    ') + "\n</svg>\n"


def cstmGenerateItemSvg(item, nodes=None, root=None, options={}):
    # Color plots need to be prepared for SVG export
    refItem = item
    if isinstance(refItem, pg.PlotItem):
        if hasattr(refItem, 'prepareForExport'):
            refItem.prepareForExport()
            app = QtCore.QCoreApplication.instance()
            app.processEvents()

    if nodes is None:  ## nodes maps all node IDs to their XML element. 
                       ## this allows us to ensure all elements receive unique names.
        nodes = {}
        
    if root is None:
        root = item
                
    ## Skip hidden items
    if hasattr(item, 'isVisible') and not item.isVisible():
        # Reset color plots back to previous values after export is complete
        if isinstance(refItem, pg.PlotItem):
            if hasattr(refItem, 'prepareForExport'):
                refItem.resetAfterExport()

        return None
        
    ## If this item defines its own SVG generator, use that.
    if hasattr(item, 'generateSvg'):
        # Reset color plots back to previous values after export is complete
        if isinstance(refItem, pg.PlotItem):
            if hasattr(refItem, 'prepareForExport'):
                refItem.resetAfterExport()
        return item.generateSvg(nodes)
    

    ## Generate SVG text for just this item (exclude its children; we'll handle them later)
    tr = QtGui.QTransform()
    if isinstance(item, QtGui.QGraphicsScene):
        xmlStr = "<g>\n</g>\n"
        doc = xml.parseString(xmlStr)
        childs = [i for i in item.items() if i.parentItem() is None]
    elif item.__class__.paint == QtGui.QGraphicsItem.paint:
        xmlStr = "<g>\n</g>\n"
        doc = xml.parseString(xmlStr)
        childs = item.childItems()
    else:
        childs = item.childItems()
        tr = itemTransform(item, item.scene())
        
        ## offset to corner of root item
        if isinstance(root, QtGui.QGraphicsScene):
            rootPos = QtCore.QPoint(0,0)
        else:
            rootPos = root.scenePos()

        tr2 = QtGui.QTransform()
        tr2.translate(-rootPos.x(), -rootPos.y())
        tr = tr * tr2

        arr = QtCore.QByteArray()
        buf = QtCore.QBuffer(arr)
        svg = QtSvg.QSvgGenerator()
        svg.setOutputDevice(buf)
        dpi = QtGui.QDesktopWidget().logicalDpiX()
        svg.setResolution(dpi)

        p = QtGui.QPainter()
        p.begin(svg)
        if hasattr(item, 'setExportMode'):
            item.setExportMode(True, {'painter': p})
        try:
            p.setTransform(tr)
            opt = QtGui.QStyleOptionGraphicsItem()
            if item.flags() & QtGui.QGraphicsItem.ItemUsesExtendedStyleOption:
                opt.exposedRect = item.boundingRect()
            item.paint(p, opt, None)
        finally:
            p.end()

        if QT_LIB in ['PySide', 'PySide2']:
            xmlStr = str(arr)
        else:
            xmlStr = bytes(arr).decode('utf-8')
        doc = xml.parseString(xmlStr.encode('utf-8'))
        
    try:
        ## Get top-level group for this item
        g1 = doc.getElementsByTagName('g')[0]
        ## get list of sub-groups
        g2 = [n for n in g1.childNodes if isinstance(n, xml.Element) and n.tagName == 'g']
        
        defs = doc.getElementsByTagName('defs')
        if len(defs) > 0:
            defs = [n for n in defs[0].childNodes if isinstance(n, xml.Element)]
    except:
        print(doc.toxml())
        raise

    ## Get rid of group transformation matrices by applying
    ## transformation to inner coordinates
    correctCoordinates(g1, defs, item, options)
    
    ## decide on a name for this item
    baseName = item.__class__.__name__
    i = 1
    while True:
        name = baseName + "_%d" % i
        if name not in nodes:
            break
        i += 1
    nodes[name] = g1
    g1.setAttribute('id', name)
    
    ## If this item clips its children, we need to take care of that.
    childGroup = g1  ## add children directly to this node unless we are clipping
    if not isinstance(item, QtGui.QGraphicsScene):
        ## See if this item clips its children
        if int(item.flags() & item.ItemClipsChildrenToShape) > 0:
            ## Generate svg for just the path
            path = QtGui.QGraphicsPathItem(item.mapToScene(item.shape()))
            item.scene().addItem(path)
            try:
                pathNode = cstmGenerateItemSvg(path, root=root, options=options)[0].getElementsByTagName('path')[0]
                # assume <defs> for this path is empty.. possibly problematic.
            finally:
                item.scene().removeItem(path)
            
            ## and for the clipPath element
            clip = name + '_clip'
            clipNode = g1.ownerDocument.createElement('clipPath')
            clipNode.setAttribute('id', clip)
            clipNode.appendChild(pathNode)
            g1.appendChild(clipNode)
            
            childGroup = g1.ownerDocument.createElement('g')
            childGroup.setAttribute('clip-path', 'url(#%s)' % clip)
            g1.appendChild(childGroup)
            
    ## Add all child items as sub-elements.
    childs.sort(key=lambda c: c.zValue())
    for ch in childs:
        csvg = cstmGenerateItemSvg(ch, nodes, root, options=options)
        if csvg is None:
            continue
        cg, cdefs = csvg
        childGroup.appendChild(cg)  ### this isn't quite right--some items draw below their parent (good enough for now)
        defs.extend(cdefs)

    # Reset color plots back to previous values after export is complete
    if isinstance(refItem, pg.PlotItem):
        if hasattr(refItem, 'prepareForExport'):
            refItem.resetAfterExport()

    return g1, defs


def correctCoordinates(node, defs, item, options):
    ## Remove transformation matrices from <g> tags by applying matrix to coordinates inside.
    ## Each item is represented by a single top-level group with one or more groups inside.
    ## Each inner group contains one or more drawing primitives, possibly of different types.
    groups = node.getElementsByTagName('g')
    
    ## Since we leave text unchanged, groups which combine text and non-text primitives must be split apart.
    ## (if at some point we start correcting text transforms as well, then it should be safe to remove this)
    groups2 = []
    for grp in groups:
        subGroups = [grp.cloneNode(deep=False)]
        textGroup = None
        for ch in grp.childNodes[:]:
            if isinstance(ch, xml.Element):
                if textGroup is None:
                    textGroup = ch.tagName == 'text'
                if ch.tagName == 'text':
                    if textGroup is False:
                        subGroups.append(grp.cloneNode(deep=False))
                        textGroup = True
                else:
                    if textGroup is True:
                        subGroups.append(grp.cloneNode(deep=False))
                        textGroup = False
            subGroups[-1].appendChild(ch)
        groups2.extend(subGroups)
        for sg in subGroups:
            node.insertBefore(sg, grp)
        node.removeChild(grp)
    groups = groups2
        
    
    for grp in groups:
        matrix = grp.getAttribute('transform')
        match = re.match(r'matrix\((.*)\)', matrix)
        if match is None:
            vals = [1,0,0,1,0,0]
        else:
            vals = [float(a) for a in match.groups()[0].split(',')]
        tr = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]]])

        removeTransform = False
        for ch in grp.childNodes:
            if not isinstance(ch, xml.Element):
                continue
            if ch.tagName == 'polyline':
                removeTransform = True
                coords = np.array([[float(a) for a in c.split(',')] for c in ch.getAttribute('points').strip().split(' ')])
                coords = pg.transformCoordinates(tr, coords, transpose=True)
                ch.setAttribute('points', ' '.join([','.join([str(a) for a in c]) for c in coords]))
            elif ch.tagName == 'path':
                removeTransform = True
                newCoords = ''
                oldCoords = ch.getAttribute('d').strip()
                if oldCoords == '':
                    continue
                for c in oldCoords.split(' '):
                    x,y = c.split(',')
                    if x[0].isalpha():
                        t = x[0]
                        x = x[1:]
                    else:
                        t = ''
                    nc = pg.transformCoordinates(tr, np.array([[float(x),float(y)]]), transpose=True)
                    newCoords += t+str(nc[0,0])+','+str(nc[0,1])+' '
                # If coords start with L instead of M, then the entire path will not be rendered.
                # (This can happen if the first point had nan values in it--Qt will skip it on export)
                if newCoords[0] != 'M':
                    newCoords = 'M' + newCoords[1:]
                ch.setAttribute('d', newCoords)
            elif ch.tagName == 'text':
                removeTransform = False
                ## leave text alone for now. Might need this later to correctly render text with outline.

                ## Correct some font information
                families = ch.getAttribute('font-family').split(',')
                if len(families) == 1:
                    font = QtGui.QFont(families[0].strip('" '))
                    if font.style() == font.SansSerif:
                        families.append('sans-serif')
                    elif font.style() == font.Serif:
                        families.append('serif')
                    elif font.style() == font.Courier:
                        families.append('monospace')
                    ch.setAttribute('font-family', ', '.join([f if ' ' not in f else '"%s"'%f for f in families]))
                
            ## correct line widths if needed
            if removeTransform and ch.getAttribute('vector-effect') != 'non-scaling-stroke' and grp.getAttribute('stroke-width') != '':
                w = float(grp.getAttribute('stroke-width'))
                s = pg.transformCoordinates(tr, np.array([[w,0], [0,0]]), transpose=True)
                w = ((s[0]-s[1])**2).sum()**0.5
                ch.setAttribute('stroke-width', str(w))
            
            # Remove non-scaling-stroke if requested
            if options.get('scaling stroke') is True and ch.getAttribute('vector-effect') == 'non-scaling-stroke':
                ch.removeAttribute('vector-effect')

        if removeTransform:
            grp.removeAttribute('transform')

def itemTransform(item, root):
    ## Return the transformation mapping item to root
    ## (actually to parent coordinate system of root)
    
    if item is root:
        tr = QtGui.QTransform()
        tr.translate(*item.pos())
        tr = tr * item.transform()
        return tr
        
    
    if int(item.flags() & item.ItemIgnoresTransformations) > 0:
        pos = item.pos()
        parent = item.parentItem()
        if parent is not None:
            pos = itemTransform(parent, root).map(pos)
        tr = QtGui.QTransform()
        tr.translate(pos.x(), pos.y())
        tr = item.transform() * tr
    else:
        ## find next parent that is either the root item or 
        ## an item that ignores its transformation
        nextRoot = item
        while True:
            nextRoot = nextRoot.parentItem()
            if nextRoot is None:
                nextRoot = root
                break
            if nextRoot is root or int(nextRoot.flags() & nextRoot.ItemIgnoresTransformations) > 0:
                break
        
        if isinstance(nextRoot, QtGui.QGraphicsScene):
            tr = item.sceneTransform()
        else:
            tr = itemTransform(nextRoot, root) * item.itemTransform(nextRoot)[0]
    
    return tr

def cleanXml(node):
    ## remove extraneous text; let the xml library do the formatting.
    hasElement = False
    nonElement = []
    for ch in node.childNodes:
        if isinstance(ch, xml.Element):
            hasElement = True
            cleanXml(ch)
        else:
            nonElement.append(ch)
    
    if hasElement:
        for ch in nonElement:
            node.removeChild(ch)
    elif node.tagName == 'g':  ## remove childless groups
        node.parentNode.removeChild(node)

pg.exporters.SVGExporter.export = cstmSVGExport

# Modified ViewBoxMenu's function to set strings that represent ranges
# as floats instead of rewriting them in scientific notation since precision
# is lost
from pyqtgraph import ViewBox # Need to import ViewBox into namespace first
def vbMenu_UpdateState(self):
    ## Something about the viewbox has changed; update the menu GUI
    state = self.view().getState(copy=False)
    if state['mouseMode'] == ViewBox.PanMode:
        self.mouseModes[0].setChecked(True)
    else:
        self.mouseModes[1].setChecked(True)

    for i in [0,1]:  # x, y
        tr = state['targetRange'][i]
        self.ctrl[i].minText.setText("%0.5f" % tr[0])
        self.ctrl[i].maxText.setText("%0.5f" % tr[1])
        if state['autoRange'][i] is not False:
            self.ctrl[i].autoRadio.setChecked(True)
            if state['autoRange'][i] is not True:
                self.ctrl[i].autoPercentSpin.setValue(state['autoRange'][i]*100)
        else:
            self.ctrl[i].manualRadio.setChecked(True)
        self.ctrl[i].mouseCheck.setChecked(state['mouseEnabled'][i])

        ## Update combo to show currently linked view
        c = self.ctrl[i].linkCombo
        c.blockSignals(True)
        try:
            view = state['linkedViews'][i]  ## will always be string or None
            if view is None:
                view = ''

            ind = c.findText(view)

            if ind == -1:
                ind = 0
            c.setCurrentIndex(ind)
        finally:
            c.blockSignals(False)

        self.ctrl[i].autoPanCheck.setChecked(state['autoPan'][i])
        self.ctrl[i].visibleOnlyCheck.setChecked(state['autoVisibleOnly'][i])
        xy = ['x', 'y'][i]
        self.ctrl[i].invertCheck.setChecked(state.get(xy+'Inverted', False))

    self.valid = True

pg.ViewBoxMenu.ViewBoxMenu.updateState = vbMenu_UpdateState

from pyqtgraph.parametertree import Parameter
class PDFExporter(pg.exporters.Exporter):
    Name = "PDF Document"
    allowCopy = False

    def __init__(self, item):
        pg.exporters.Exporter.__init__(self, item)

        # Orientation option
        self.params = Parameter(name='params', type='group', children=[
            {'name':'Orientation: ', 'type':'list', 'values':['Portrait',
                'Landscape']}
        ])

    def parameters(self):
        return self.params

    def export(self, fileName=None, toBytes=False, copy=False):
        # Get a filename and add extension if it is missing
        if fileName is None and not toBytes and not copy:
            self.fileSaveDialog(filter='*.pdf')
            return

        if '.pdf' not in fileName:
            fileName = f"{fileName}.pdf"

        # Initialize PDF Writer paint device
        self.pdfFile = QtGui.QPdfWriter(fileName)

        # Set page orientation if user selected 'Landscape' mode
        horzLt = self.params['Orientation: '] == 'Landscape'
        if horzLt:
            self.pdfFile.setPageOrientation(QtGui.QPageLayout.Landscape)

        # Get the device resolution and set resolution for the PDF Writer
        res = QtGui.QDesktopWidget().logicalDpiX()
        self.pdfFile.setResolution(res)

        # Get the paintRect for the page in pixels
        pageLt = self.pdfFile.pageLayout()
        targetRect = pageLt.paintRectPixels(res)

        ## Map to QRectF and remove margins
        targetRect = QtCore.QRectF(targetRect)
        margins = QtCore.QMarginsF(pageLt.marginsPixels(res))
        targetRect.moveTopLeft(QtCore.QPointF(0, 0))
        targetRect = targetRect.marginsRemoved(margins)

        # Get the source rect
        sourceRect = self.getSourceRect()

        # Start painter and render scene
        painter = QtGui.QPainter(self.pdfFile)
        try:
            self.setExportMode(True, {'painter': painter})
            self.getScene().render(painter, targetRect, QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()

# Add PDF exporter to list of exporters
PDFExporter.register()