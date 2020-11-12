import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
from .timeManager import TimeManager
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
from dateutil import rrule
from dateutil.rrule import rrule
from dateutil.rrule import rrule, SECONDLY, MINUTELY, HOURLY, DAILY, MONTHLY, YEARLY

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
    
    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        ''' Calls dataBounds() on each plot data item
            in plot and returns the min/max upper bounds
            for the plot if valid
        '''
        pdis = self.listDataItems()
        plot_min = None
        plot_max = None
        for pdi in pdis:
            lower, upper = pdi.dataBounds(ax, frac, orthoRange)
            if lower is not None and not np.isnan(lower):
                if plot_min is None:
                    plot_min = lower
                else:
                    plot_min = min(lower, plot_min)

                if plot_max is None:
                    plot_max = upper
                else:
                    plot_max = max(upper, plot_max)

        return (plot_min, plot_max)

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
        if (ev.button() == QtCore.Qt.LeftButton):
            self.axisClicked.emit()

class DateAxis(pg.AxisItem):
    ticksChanged = QtCore.pyqtSignal(object)
    axisClicked = QtCore.pyqtSignal()
    formatChanged = QtCore.pyqtSignal(str)
    def __init__(self, epoch, orientation, offset=0, *args, **kwargs):
        self.tickOffset = offset
        self.timeRange = None
        self.epoch = epoch
        self.levels = None
        self.tm = TimeManager(0, 0, self.epoch)
        self.label_visible = False # Keeps track of whether label is visible

        super().__init__(orientation, *args, **kwargs)
        self.enableAutoSIPrefix(False)

        # Time mode information
        self.order = ['ms', 'sec', 'min', 'hours', 'days', 'months', 'years']
        self.factors = { # Factor to multiply units by to convert to seconds
            'ms' : 1,
            'sec' : 1,
            'min' : 60,
            'hours': 60 ** 2,
            'days' : timedelta(days=1).total_seconds(),
            'months' : timedelta(days=30).total_seconds(),
            'years' : timedelta(days=365).total_seconds()
        }

        self.modes = { # Default label format for each mode
            'ms' : 'SS.SSSSS',
            'sec' : 'MM:SS',
            'min' : 'HH:MM:SS',
            'hours' : 'HH:MM',
            'days' : 'DATE HH:MM',
            'months' : 'DATE',
            'years' : 'DATE'
        }

        self.rule_keys = { # Maps time mode to a frequency key for rrule
            'sec' : SECONDLY,
            'min' : MINUTELY,
            'hours' : HOURLY,
            'days' : DAILY,
            'months' : MONTHLY,
            'years' : YEARLY,
        }

        self.min_ticks = 4 # Minimum number of ticks

        # Lower bounds for time length or each time mode (in seconds)
        self.bases = {} 
        for key in self.order:
            self.bases[key] = timedelta(seconds=self.factors[key]*self.min_ticks)

        # Fixed set of spacers for selected time modes
        self.spacers = {
            'months' : [1, 2, 3, 6, 12],
            'hours' : [1, 2, 3, 6, 12],
            'min' : [1, 2, 5, 10, 15, 30],
            'sec' : [1, 2, 5, 10, 15, 30],
            'ms' : [0.1, 0.2, 0.5, 1, 2, 5],
        }
        # Custom tick spacing
        self.tickDiff = None

        # Custom label formatting
        self.lbl_fmt = None
        self.curr_mode = None

        fmts = ['DATE', 'DATE HH:MM', 'DATE HH:MM:SS', 'HH',
            'HH:MM', 'HH:MM:SS', 'MM', 'MM:SS', 'MM:SS.SSS', 'SS.SSSSS']

        format_modes = ['days', 'min', 'sec', 'hours', 
            'min', 'sec', 'min', 'sec', 'ms', 'ms']

        date = '%Y %b %d'
        format_strs = [date, f'{date} %H:%M', f'{date} %H:%M:%S', '%H',
            '%H:%M', '%H:%M:%S', '%M', '%M:%S', '%M:%S.%f', '%S.%f']
        
        # Create dictionaries for each label format --> mode and formatting string
        self.fmt_modes = {}
        self.fmt_strs = {}
        for fmt, fmt_mode, fmt_str in zip(fmts, format_modes, format_strs):
            self.fmt_modes[fmt] = fmt_mode
            self.fmt_strs[fmt] = fmt_str

    def getTickFont(self):
        if hasattr(self, 'tickFont'):
            return self.tickFont
        else:
            return self.style['tickFont']

    def resetLabelFormat(self):
        self.set_label_fmt(None)
        self.picture = None
        self.update()

    def setRange(self, mn, mx):
        super().setRange(mn, mx)
        self.tm.tO = mn + self.tickOffset
        self.tm.tE = mx + self.tickOffset

    def get_label_modes(self):
        ''' Returns the full list of label formats available '''
        return list(self.fmt_strs.keys())

    def get_range(self):
        ''' Returns the range of the axis item '''
        return (self.tm.tO, self.tm.tE)

    def get_default_mode(self):
        ''' Returns the time mode set for the axis item '''
        if self.curr_mode is None:
            # If no current time mode is set, get the time
            # range delta and guess mode
            minVal, maxVal = self.get_range()
            start = self.tm.getDateTimeObjFromTick(minVal)
            end = self.tm.getDateTimeObjFromTick(maxVal)
            td = abs(end-start)
            mode = self.guess_mode(td)
        else:
            mode = self.curr_mode
        return mode
    
    def set_label_fmt(self, fmt):
        ''' Sets the label format for the axis item '''
        old_fmt = self.lbl_fmt

        # Set label format and reset curr_mode
        self.lbl_fmt = fmt
        self.curr_mode = None

        # Update text if visible
        if self.label_visible:
            label = self.get_label()
            self.setLabel(label)

        # Reset image
        self.picture = None
        self.update()

        # Emit signal if format changed
        if old_fmt != fmt:
            self.formatChanged.emit(fmt)

    def get_label_fmt(self):
        ''' Returns label format value, None if not set '''
        return self.lbl_fmt

    def showLabel(self, show=True):
        super().showLabel(show)
        self.label_visible = show

    def get_label(self):
        ''' Returns the label format '''
        if self.lbl_fmt:
            return self.lbl_fmt
        else: # Default time label
            return self.modes[self.get_default_mode()]

    def setLabelVisible(self, val):
        if val:
            label = self.get_label()
            self.setLabel(label)
        else:
            self.showLabel(False)

    def guess_mode(self, td):
        ''' Get the default time mode by comparing
            the timedelta to the lower bounds for the
            time range for each time mode
        '''
        mode = 'ms'
        for key in self.order[::-1]:
            if td > self.bases[key]:
                mode = key
                break
        
        return mode

    def get_default_spacer(self, td, mode, num_ticks=6):
        ''' Returns the default spacer values for the
            time range length and time mode
        '''

        # Get spacer values to search through
        if mode in self.spacers:
            bins = self.spacers[mode]
        else:
            # Get bins in [1**i, 2**i, 5**i] format,
            # similar to original algorithm
            bins = []
            upper = np.ceil(np.log10(td.total_seconds()))
            for i in range(0, int(upper)):
                level = [n * (10**i) for n in [1,2,5]]
                bins.extend(level)

        # Estimate timedelta for ticks, map to time mode units,
        # and get nearest value in bin
        frac = td / num_ticks
        num = np.round(frac.total_seconds() / self.factors[mode])
        index = bisect_left(bins, num)
        if index >= len(bins):
            index = len(bins) - 1

        # Map bin value back to time
        spacer_vals = [bins[index]]
        if index >= 2:
            lower = index - 2
            # Map odd-indexed spacer's minor values to
            # the previous index's spacer
            if (index + 1) % 2 == 0:
                lower = index - 1
            spacer_vals.append(bins[lower])
        elif index == 1:
            spacer_vals.append(bins[0])

        return spacer_vals

    def get_tick_dates(self, start, end, mode, spacer, cstm=False):
        # Get 'floor' date relative to start date and
        # calculate difference
        start_day = start.date()
        start_day = datetime.combine(start_day, datetime.min.time())
        start_diff = start - start_day

        # Map spacer to seconds
        if cstm:
            spacer_secs = spacer
        else:
            spacer_secs = spacer * self.factors[mode]

        # Get number of spacer values need to round up to
        # nearest 'whole' time in spacer quantities
        num_to_start = start_diff.total_seconds() / spacer_secs
        num_to_start = np.ceil(num_to_start)
        clean_start = start_day + num_to_start * timedelta(seconds=spacer_secs)

        # Monthly spacers do not work properly with seconds calculation
        if mode == 'months':
            clean_start = datetime(start.year, start.month, 1)

        # Get tick dates using rrule from dateutil
        if mode != 'ms' and not cstm:
            key = self.rule_keys[mode]
            dates = rrule(key, dtstart=clean_start, until=end, interval=spacer)
            dates = list(dates)
        else:
            dates = []
            num_dates = int(abs(end-start).total_seconds() / spacer)
            for i in range(0, num_dates):
                dates.append(clean_start + timedelta(seconds=spacer)*i)
        
        return dates

    def tickValues(self, minVal, maxVal, size):
        # Map to timedelta
        start = self.tm.getDateTimeObjFromTick(minVal+self.tickOffset)
        end = self.tm.getDateTimeObjFromTick(maxVal+self.tickOffset)
        td = abs(end-start)

        # Get label format mode and spacer values
        lbl_str = ''
        if self.lbl_fmt is None:
            mode = self.guess_mode(td)
            lbl_str = self.modes[mode]
        else:
            # If using a custom label, guess the mode from the tick range
            # and then use the larger of the two quantities
            tick_mode = self.guess_mode(td)
            label_mode = self.fmt_modes[self.lbl_fmt]
            if self.order.index(tick_mode) < self.order.index(label_mode):
                mode = label_mode
            else:
                mode = tick_mode

            lbl_str = self.lbl_fmt
        self.curr_mode = mode

        # Get tick spacings
        if self.tickDiff:
            spacer_vals = [int(diff.total_seconds()) for diff in self.tickDiff]
            cstm = True
        else:
            spacer_vals = self.get_default_spacer(td, mode)
            cstm = False

        # Get tick dates for each spacer group and map
        # to tick values
        vals = []
        seen = []
        for spacer in spacer_vals:
            # Get tick dates in range with given spacer and map to ticks
            dates = self.get_tick_dates(start, end, mode, spacer, cstm)
            ticks = [self.tm.getTickFromDateTime(date) for date in dates]
            ticks = [tick - self.tickOffset for tick in ticks]

            # Remove previously seen ticks
            unique_ticks = []
            for t in ticks:
                if t not in seen:
                    unique_ticks.append(t)
            ticks = unique_ticks

            # Add ticks to list if ticks are not empty
            if len(ticks) > 0:
                vals.append((0, ticks))
                seen.extend(ticks)

        return vals
    
    def get_fmt_str(self):
        ''' Returns formatting string used to convert datetime
            to a timestamp
        '''
        fmt_str = ''
        if self.lbl_fmt is None:
            # Use default label format
            lbl_fmt = self.modes[self.curr_mode]
            fmt_str = self.fmt_strs[lbl_fmt]
        else:
            # Map set label format to formatting string
            fmt_str = self.fmt_strs[self.lbl_fmt]

        return fmt_str

    def tickStrings(self, values, scale, spacing):
        # Get formatting string
        fmt_str = self.get_fmt_str()

        # Map each tick value to a datetime object
        # and use strftime to format into a string
        strs = []
        for value in values:
            value = value + self.tickOffset
            date = self.tm.getDateTimeObjFromTick(value)
            ts = date.strftime(fmt_str)
            strs.append(ts)

        return strs

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
        if (ev.button() == QtCore.Qt.LeftButton):
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