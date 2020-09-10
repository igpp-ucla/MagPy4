import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
from .timeManager import TimeManager
from datetime import datetime, timedelta
from FF_Time import FFTIME
from bisect import bisect_left, bisect_right

class MagPyPlotItem(pg.PlotItem):
    def __init__(self, epoch=None, name=None, selectable=False, *args, **kwargs):
        # Initialize axis items
        axisKws = ['left', 'top', 'right', 'bottom']
        axisItems = {kw:MagPyAxisItem(kw) for kw in axisKws}

        # If epoch is passed, set bottom/top axes to Date Axes
        self.epoch = epoch
        if epoch:
            axisItems['bottom'] = DateAxis(epoch, 'bottom')
            axisItems['top'] = DateAxis(epoch, 'top')

        # Update axis items if any custom ones were passed
        if 'axisItems' in kwargs:
            axisItems.update(kwargs['axisItems'])
            del kwargs['axisItems']

        # Set default name, None if none passed
        self.name = name
        self.varInfo = []
        self.plotApprAction = None
        self.plotAppr = None

        pg.PlotItem.__init__(self, axisItems=axisItems, *args, **kwargs)
        self.hideButtons()

    def isSpecialPlot(self):
        '''
            Returns whether this is regular line plot or not
        '''
        return False
    
    def setName(self, name):
        ''' Sets internal name/label for this plot '''
        self.name = name
    
    def getName(self, name):
        ''' Returns internal name/label for this plot '''
        return self.name

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

    def plot(self, *args, **kargs):
        clear = kargs.get('clear', False)
        params = kargs.get('params', None)
          
        if clear:
            self.clear()
            
        item = MagPyPlotDataItem(*args, **kargs)

        if params is None:
            params = {}

        self.addItem(item, params=params)
        
        return item
    
    def getLineInfo(self):
        '''
            Extract dictionaries containing info about
            the plot data item, trace pen, and name
        '''
        if self.isSpecialPlot():
            return []

        # Save info for each plot data item
        infos = []
        pdis = self.listDataItems()
        for pdi in pdis:
            info = {}
            info['pen'] = pdi.opts['pen']
            info['item'] = pdi
            info['name'] = pdi.name()
            infos.append(info)
        
        return infos
    
    def setVarInfo(self, info):
        self.varInfo = info
    
    def enablePlotAppr(self, val):
        if val:
            self.plotApprAction = QtWidgets.QAction('Change Plot Appearance...')
            self.plotApprAction.triggered.connect(self.openPlotAppr)
        else:
            self.plotApprAction = None
    
    def getContextMenus(self, ev):
        menu = self.getMenu()
        if self.plotApprAction:
            return [self.plotApprAction, menu]
        else:
            return menu

    def openPlotAppr(self):
        from .plotAppearance import PlotAppearance
        self.plotAppr = PlotAppearance(self, [self])
        self.plotAppr.show()

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
    textSizeChanged = QtCore.pyqtSignal(object)
    axisClicked = QtCore.pyqtSignal()
    def __init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True):
        self.tickDiff = None
        self.minWidthSet = False
        self.levels = None
        pg.AxisItem.__init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True)

    def getTickSpacing(self):
        return self.tickDiff

    def getTickFont(self):
        if hasattr(self, 'tickFont'):
            return self.tickFont
        else:
            return self.style['tickFont']

    def setCstmTickSpacing(self, diff):
        ''' Sets tick spacing to major/minor values in diff '''
        if diff is None:
            self.resetTickSpacing()
            return

        # If diff is a list, extract major and minor values
        if isinstance(diff, (list, np.ndarray)):
            if len(diff) > 1:
                major, minor = diff[0], diff[1]
                diff = [major, minor]
            else:
                major = diff[0]
                minor = major * 1000
                diff = [major]
        else:
            # Otherwise, make a list with just major in it
            # and set minor to some very large value
            major = diff
            minor = major * 1000
            diff = [major]

        self.setTickSpacing(major=major, minor=minor)
        self.tickDiff = diff

    def resetTickSpacing(self):
        self._tickSpacing = None
        self.picture = None
        self.update()

    def setLogMode(self, log):
        self.logMode = log
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        return pg.AxisItem.tickStrings(self, values, scale, spacing)

    def setLogMode(self, val):
        pg.AxisItem.setLogMode(self, val)
        if val:
            self.setCstmTickSpacing(1)
        else:
            self.resetTickSpacing()

    def axisType(self):
        return 'Regular' if not self.logMode else 'Log'

    def _updateMaxTextSize(self, x):
        ## Informs that the maximum tick size orthogonal to the axis has
        ## changed; we use this to decide whether the item needs to be resized
        ## to accomodate.
        if self.orientation in ['left', 'right']:
            mx = max(self.textWidth, x)
            if mx > self.textWidth or mx < self.textWidth-10 or not self.minWidthSet:
                self.textWidth = mx
                if self.style['autoExpandTextSpace'] is True:
                    self._updateWidth()
                self.minWidthSet = True
        else:
            mx = max(self.textHeight, x)
            if mx > self.textHeight or mx < self.textHeight-10:
                self.textHeight = mx
                if self.style['autoExpandTextSpace'] is True:
                    self._updateHeight()

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
        if self.logMode:
            w += 10
        return w # 10 extra to offset if in scientific notation

    def logTickStrings(self, values, scale, spacing):
        return [f'10<sup>{int(x)}</sup>' for x in np.array(values).astype(float)]

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)

        self.drawAxis(p, tickSpecs, axisSpec)
        self.drawText(p, textSpecs)

    def drawAxis(self, p, tickSpecs, axisSpec):
        ''' Draw axis lines and tick markers '''
        # Draw long axis line
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)
        
        # Draw tick markers
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

    def drawText(self, p, textSpecs):
        ''' Paint tick labels '''
        # Set pen font
        if self.getTickFont() is not None:
            p.setFont(self.getTickFont())

        # Draw labels in rich text
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            qst = QtGui.QStaticText(text)
            qst.setTextFormat(QtCore.Qt.RichText)
            p.drawStaticText(rect.left(), rect.top(), qst)

    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be 
        interpreted by drawPicture().
        """
        if self.getTickFont():
            p.setFont(self.getTickFont())
        visibleLevels = []

        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = pg.Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        tickPositions = [] # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i*0.5)+1.0)

            lineAlpha = 255 / (i+1)
            if self.grid is not False:
                lineAlpha *= self.grid/255. * np.clip((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)
            
            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)
                
                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(lineAlpha)
                tickPen.setColor(color)
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))

        if self.style['stopAxisAtTick'][0] is True:
            stop = max(span[0].y(), min(map(min, tickPositions)))
            if axis == 0:
                span[0].setY(stop)
            else:
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            stop = min(span[1].y(), max(map(max, tickPositions)))
            if axis == 0:
                span[1].setY(stop)
            else:
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text

        textSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel']+1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            spacing, values = tickLevels[i]
            if self.logMode and spacing is None:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, pg.asUnicode(s))
                    if self.logMode:
                        # Calculate bounding rect for rich-text log tick labels
                        txtItem = QtWidgets.QGraphicsTextItem()
                        txtItem.setHtml(s)
                        txtItem.setFont(p.font())
                        txt_br = txtItem.boundingRect()
                        width = txt_br.width() * 0.8
                        height = txt_br.height()
                        lft = 50 - (width/2)
                        top = 50 - (height/2)
                        br = QtCore.QRectF(lft, top, width, height)
                    
                    br.setHeight(br.height() * 0.8)
                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            # Determine exactly where tick text should be drawn
            visibleRow = []
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                vstr = pg.asUnicode(vstr)
                x = tickPositions[i][j]
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                #self.textHeight = height
                offset = max(0,self.style['tickLength']) + textOffset
                if self.orientation == 'left':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop-offset-width, x-(height/2), width, height)
                elif self.orientation == 'right':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop+offset, x-(height/2), width, height)
                elif self.orientation == 'top':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                    rect = QtCore.QRectF(x-width/2., tickStop-offset-height, width, height)
                elif self.orientation == 'bottom':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                    rect = QtCore.QRectF(x-width/2., tickStop+offset, width, height)

                textSpecs.append((rect, textFlags, vstr))
                visibleRow.append(tickLevels[i][1][j])
            
            visibleLevels.append((tickLevels[i][0], visibleRow))

        ## update max text size if needed.
        self._updateMaxTextSize(textSize2)
        self.levels = visibleLevels

        return (axisSpec, tickSpecs, textSpecs)

    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
        self.axisClicked.emit()

class DateAxis(pg.AxisItem):
    ticksChanged = QtCore.pyqtSignal(object)
    axisClicked = QtCore.pyqtSignal()
    def __init__(self, epoch, orientation, offset=0, *args, **kwargs):
        self.tickOffset = offset
        self.timeRange = None
        self.epoch = epoch
        self.levels = None
        self.tm = TimeManager(0, 0, self.epoch)
        pg.AxisItem.__init__(self, orientation,  *args, **kwargs)
        self.enableAutoSIPrefix(False)

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

    def getTickFont(self):
        if hasattr(self, 'tickFont'):
            return self.tickFont
        else:
            return self.style['tickFont']

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
            seconds_diff = []
            for val in self.tickDiff:
                seconds_diff.append((val.total_seconds(), 0))
            return seconds_diff
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

    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be 
        interpreted by drawPicture().
        """
        if self.getTickFont():
            p.setFont(self.getTickFont())
        visibleLevels = []

        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = pg.Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        tickPositions = [] # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i*0.5)+1.0)

            lineAlpha = 255 / (i+1)
            if self.grid is not False:
                lineAlpha *= self.grid/255. * np.clip((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)
            
            for v in ticks:
                ## determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  ## last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)
                
                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength*tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(lineAlpha)
                tickPen.setColor(color)
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))

        if self.style['stopAxisAtTick'][0] is True:
            stop = max(span[0].y(), min(map(min, tickPositions)))
            if axis == 0:
                span[0].setY(stop)
            else:
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            stop = min(span[1].y(), max(map(max, tickPositions)))
            if axis == 0:
                span[1].setY(stop)
            else:
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text

        textSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel']+1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            spacing, values = tickLevels[i]
            if self.logMode and spacing is None:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, pg.asUnicode(s))
                    if self.logMode:
                        # Calculate bounding rect for rich-text log tick labels
                        txtItem = QtWidgets.QGraphicsTextItem()
                        txtItem.setHtml(s)
                        txtItem.setFont(p.font())
                        txt_br = txtItem.boundingRect()
                        width = txt_br.width() * 0.8
                        height = txt_br.height()
                        lft = 50 - (width/2)
                        top = 50 - (height/2)
                        br = QtCore.QRectF(lft, top, width, height)
                    
                    br.setHeight(br.height() * 0.8)
                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            # Determine exactly where tick text should be drawn
            visibleRow = []
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                vstr = pg.asUnicode(vstr)
                x = tickPositions[i][j]
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                #self.textHeight = height
                offset = max(0,self.style['tickLength']) + textOffset
                if self.orientation == 'left':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop-offset-width, x-(height/2), width, height)
                elif self.orientation == 'right':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop+offset, x-(height/2), width, height)
                elif self.orientation == 'top':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignBottom
                    rect = QtCore.QRectF(x-width/2., tickStop-offset-height, width, height)
                elif self.orientation == 'bottom':
                    textFlags = QtCore.Qt.TextDontClip|QtCore.Qt.AlignCenter|QtCore.Qt.AlignTop
                    rect = QtCore.QRectF(x-width/2., tickStop+offset, width, height)

                textSpecs.append((rect, textFlags, vstr))
                visibleRow.append(tickLevels[i][1][j])

            visibleLevels.append((tickLevels[i][0], visibleRow))

        ## update max text size if needed.
        self._updateMaxTextSize(textSize2)
        if visibleLevels != self.levels:
            self.ticksChanged.emit(visibleLevels)
        self.levels = visibleLevels

        return (axisSpec, tickSpecs, textSpecs)

    def tickStrings(self, values, scale, spacing):
        # Convert start/end times to strings
        strings = []
        for v in values:
            ts = (FFTIME(v+self.tickOffset, Epoch=self.tm.epoch)).UTC
            s = self.fmtTimeStmp(ts)
            strings.append(s)
        return strings

    def setCstmTickSpacing(self, diff):
        if isinstance(diff, (list, np.ndarray)):
            self.tickDiff = diff
        else:
            self.tickDiff = [diff]
        self.picture = None
        self.update()

    def resetTickSpacing(self):
        self.tickDiff = None
        self.picture = None
        self.update()

    def axisType(self):
        return 'DateTime'

    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
        self.axisClicked.emit()

class MagPyPlotDataItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self.connectSegments = None
        pg.PlotDataItem.__init__(self, *args, **kwargs)

        if isinstance(self.opts['connect'], (list, np.ndarray)):
            self.connectSegments = self.opts['connect']

    def getSegs(self, segs, ds, n=None):
        # Splits connection vals by ds, setting values to zero if there is a break
        # in between each segment
        n = int(len(segs)/ds) if n is None else n

        # Get indices where there are breaks
        zeroIndices = np.where(segs == 0)[0]

        # Split connection values by ds
        segs = segs[:n*ds:ds]

        # Find the segments from original connection list where there are breaks 
        # in and set the new connection value to zero
        for z in zeroIndices:
            i = int(z/ds)
            if i < len(segs):
                segs[i] = 0

        return segs

    def getData(self):
        if self.xData is None:
            return (None, None)

        if self.xDisp is None:
            x = self.xData
            y = self.yData

            if self.connectSegments is not None:
                segs = self.connectSegments[:]
            else:
                segs = self.connectSegments

            if self.opts['fftMode']:
                x,y = self._fourierTransform(x, y)
                # Ignore the first bin for fft data if we have a logx scale
                if self.opts['logMode'][0]:
                    x=x[1:]
                    y=y[1:]
            if self.opts['logMode'][0]:
                x = np.log10(x)
            if self.opts['logMode'][1]:
                y = np.log10(y)

            ds = self.opts['downsample']
            if not isinstance(ds, int):
                ds = 1

            if self.opts['autoDownsample']:
                range = self.viewRect()
                if range is not None:
                    dx = float(x[-1]-x[0]) / (len(x)-1)
                    x0 = (range.left()-x[0]) / dx
                    x1 = (range.right()-x[0]) / dx
                    width = self.getViewBox().width()

                    # If plot not visible yet, try using the screen width
                    if width == 0 or width < 100:
                        screenSize = QtGui.QGuiApplication.primaryScreen().geometry()
                        width = screenSize.width()

                    if width != 0.0:
                        width *= 2 # Prefer smaller downsampling factors
                        ds = int(max(1, int((x1-x0) / (width*self.opts['autoDownsampleFactor']))))

            if self.opts['clipToView']:
                view = self.getViewBox()
                if view is None or not view.autoRangeEnabled()[0]:
                    # this option presumes that x-values have uniform spacing
                    range = self.viewRect()
                    if range is not None and len(x) > 1:
                        lft = range.left()
                        rght = range.right()
                        x0 = bisect_left(x, lft)
                        x0 = max(0, x0-1)
                        x1 = bisect_right(x, rght)
                        x = x[x0:x1+1]
                        y = y[x0:x1+1]
                        if self.connectSegments is not None:
                            segs = segs[x0:x1+1]

            if ds > 1:
                if self.opts['downsampleMethod'] == 'subsample':
                    x = x[::ds]
                    y = y[::ds]
                    if self.connectSegments is not None:
                        segs = list(self.getSegs(segs, ds))

                elif self.opts['downsampleMethod'] == 'mean':
                    n = len(x) // ds
                    x = x[:n*ds:ds]
                    y = y[:n*ds].reshape(n,ds).mean(axis=1)

                    if self.connectSegments is not None:
                        segs = np.array(list(self.getSegs(segs, ds, n)))

                elif self.opts['downsampleMethod'] == 'peak':
                    n = len(x) // ds
                    # Reshape x values
                    x1 = np.empty((n,2))
                    x1[:] = x[:n*ds:ds,np.newaxis]
                    x = x1.reshape(n*2)

                    # Reshape y values
                    y1 = np.empty((n,2))
                    y2 = y[:n*ds].reshape((n, ds))
                    y1[:,0] = y2.max(axis=1)
                    y1[:,1] = y2.min(axis=1)
                    y = y1.reshape(n*2)

                    # Reshape connection list
                    if self.connectSegments is not None:
                        segs = np.array(list(self.getSegs(segs, ds, n)))
                        segs_new = np.empty((n, 2))
                        segs_new[:] = segs[::,np.newaxis]
                        segs = segs_new.reshape(n*2)

            self.xDisp = x
            self.yDisp = y

            if self.connectSegments is not None:
                self.opts['connect'] = segs

        return self.xDisp, self.yDisp

    def updateItems(self):
        x, y = self.getData()
        curveArgs = {}
        for k,v in [('pen','pen'), ('shadowPen','shadowPen'), ('fillLevel','fillLevel'), ('fillBrush', 'brush'), ('antialias', 'antialias'), ('connect', 'connect'), ('stepMode', 'stepMode')]:
            curveArgs[v] = self.opts[k]

        scatterArgs = {}
        for k,v in [('symbolPen','pen'), ('symbolBrush','brush'), ('symbol','symbol'), ('symbolSize', 'size'), ('data', 'data'), ('pxMode', 'pxMode'), ('antialias', 'antialias')]:
            if k in self.opts:
                scatterArgs[v] = self.opts[k]

        if curveArgs['pen'] is not None or (curveArgs['brush'] is not None and curveArgs['fillLevel'] is not None):
            self.curve.setData(x=x, y=y, **curveArgs)
            self.curve.show()
        else:
            self.curve.hide()

        if scatterArgs['symbol'] is not None:
            self.scatter.setData(x=x, y=y, **scatterArgs)
            self.scatter.show()
        else:
            self.scatter.hide()