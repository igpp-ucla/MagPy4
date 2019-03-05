
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
from FF_Time import FFTIME
from datetime import datetime, timedelta
from math import ceil
import functools

# Label item placed at top
class RegionLabel(pg.InfLineLabel):
    def __init__(self, lines, text="", movable=False, position=0.5, anchors=None, **kwds):
        self.lines = lines
        position = 0.975
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
    def __init__(self, window, plotItems, values=(0, 1), mode=None, color=None):
        pg.LinearRegionItem.__init__(self, values=(0,0))
        self.window = window
        self.plotItems = plotItems
        self.regionItems = []
        self.labelText = mode if mode is not None else ''
        self.fixedLine = False

        for plt in self.plotItems:
            # Create a LinearRegionItem for each plot with same initialized vals
            regionItem = LinkedSubRegion(self, values=values, color=color)
            regionItem.setBounds([0, self.window.maxTime-self.window.tickOffset])
            plt.addItem(regionItem)
            self.regionItems.append(regionItem)
            # Connect plot's regions/lines to each other
            line0, line1 = regionItem.lines
            line0.sigDragged.connect(functools.partial(self.linesChanged, line0, 0))
            line1.sigDragged.connect(functools.partial(self.linesChanged, line1, 1))

        # Initialize region label at top-most plot
        self.labelPltIndex = 0
        self.setLabel(self.labelPltIndex)

        # Initial connection to time edit
        self.window.connectLinesToTimeEdit(self.window.selectTimeEdit, self)

    def setLabel(self, plotNum):
        # Create new label for line
        fillColor = pg.mkColor('#212121') # Dark grey background color
        fillColor.setAlpha(200)
        opts = {'movable': False, 'position':1.0, 'color': pg.mkColor('#FFFFFF'),
            'fill': fillColor}
        label = RegionLabel(self.regionItems[plotNum].lines, text=self.labelText, **opts)
        # Update line's label and store the plot index where it's currently located
        self.regionItems[plotNum].lines[0].label = label
        self.labelPltIndex = plotNum

    def mouseDragEvent(self, ev):
        # Called by sub-regions to drag all regions at same time
        for regIt in self.regionItems:
            pg.LinearRegionItem.mouseDragEvent(regIt, ev)
        self.updateWindowInfo()

    def linesChanged(self, line, lineNum, extra):
        # Update all lines on same side of region as the original 'sub-line'
        pos = line.value()
        for regItem in self.regionItems:
            lines = regItem.lines
            lines[lineNum].setValue(pos)

            # Update other line as well if region width is fixed to 0
            if self.fixedLine:
                lines[int(not lineNum)].setValue(pos)
        # Update trace stats / timeEdits accordingly
        self.updateWindowInfo()

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
            plotIndex += 1
            # Find the next visible sub-region and place the label there
            while plotIndex < len(self.regionItems):
                if self.regionItems[plotIndex].isVisible():
                    self.setLabel(plotIndex)
                    return
                plotIndex += 1
        # If adding in a sub-region at plot index higher than the one
        # with the region label, move the label to this plot
        elif plotIndex < self.labelPltIndex and visible:
            label.setVisible(False)
            self.setLabel(plotIndex)

    def updateWindowInfo(self):
        if self.window.selectTimeEdit == None:
            return
        # Update trace stats, connect lines to time edit, and update time edit
        # values, if applicable
        self.window.connectLinesToTimeEdit(self.window.selectTimeEdit, self)
        self.window.updateTimeEditByLines(self.window.selectTimeEdit, self)
        self.window.updateTraceStats()

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

    def mouseDragEvent(self, ev):
        # If this sub-region is dragged, move all other sub-regions accordingly
        self.grp.mouseDragEvent(ev)

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
        return 'Regular'

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, customTicks, customStrings, customSpacing, *args, **kwargs):
        self.tickDiff = None
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont = QtGui.QFont()
        self.tickFont.setPixelSize(14)
        self.customStrings = customStrings
        self.customSpacing = customSpacing
        if customTicks:
            self.style['maxTextLevel'] = 1 # never have any subtick labels
            self.style['textFillLimits'] = [(0,1.1)] # try to always draw labels
        #self.style['tickLength'] = -10

    # todo: should just try to figure out range of values and add decimals if small enough
    def tickStrings(self, values, scale, spacing):
        if self.customStrings:
            return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below
        #return [f'{x:.1f}    ' for x in values]
        return pg.AxisItem.tickStrings(self,values,scale,spacing)

    def tickSpacing(self, minVal, maxVal, size):
        if self.customSpacing:
            if self.tickDiff:
                return [(self.tickDiff, 0)]
            else:
                return [(10.0,0),(1.0,0),(0.5,0)]
        return pg.AxisItem.tickSpacing(self,minVal,maxVal,size)

    def axisType(self):
        if self.customStrings:
            return 'Log'
        else:
            return 'Regular'

    # overriden from source to be able to have superscript text
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        if not self.customStrings:
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
    def __init__(self, orientation, pen=None, linkView=None, parent=None,
                maxTickLength=-5, showValues=True):
        self.tickOffset = 0
        pg.AxisItem.__init__(self, orientation, pen, linkView, parent,
                            maxTickLength,showValues)
        # Dictionary holding default increment values for ticks
        self.modeToDelta = {}
        self.modeToDelta['DAY'] = timedelta(hours=6)
        self.modeToDelta['HR'] = timedelta(minutes=30)
        self.modeToDelta['MIN'] = timedelta(minutes=5)
        self.modeToDelta['MS'] = timedelta(milliseconds=500)
        # String used by strftime/strptime to parse UTC strings
        self.fmtStr = '%Y %j %b %d %H:%M:%S.%f'
        self.tickDiff = None

    # Format a timestamp according to format underneath axis
    def fmtTimeStmp(self, window, times):
        splits = times.split(' ')
        t = splits[4]
        rng = self.window.getSelectedTimeRange()

        if rng > window.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            return f'{splits[2]} {splits[3]} {t.split(":")[0]}:{t.split(":")[1]}'
        elif rng > window.hrCutoff: # if over half hour show hh:mm:ss
            return t.rsplit('.',1)[0]
        elif rng > window.minCutoff: # if over 10 seconds show mm:ss
            return t.split(':',1)[1].split('.')[0]
        else:
            return t.split(':',1)[1] # else show mm:ss.mmm

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

    # Edge cases: First tick at rounded up time, last tick at rounded down time
    def firstTickDt(self, minDt, td):
        # Increment with timedelta until reach first tick that will be visible
        startDt = self.zeroLowerVals(minDt, td)
        while startDt < minDt:
            startDt = startDt + td
        return startDt
    def lastTickDt(self, maxDt, td):
        # Increment with timedelta until reach tick just past edge
        endDt = self.zeroLowerVals(maxDt, td)
        while endDt <= maxDt:
            endDt = endDt + td
        return endDt - td

    # From the list of all potential ticks in a range, uniformly skip over
    # certain ones so there aren't too many on a plot
    def limitTicks(self, tickList):
        tickMax = 6 # Maximum number of ticks to show on a plot
        listLen = len(tickList)
        if listLen > tickMax:
            # Only keep entries w/ indices that are multiples of skipNum
            skipNum = ceil(listLen / tickMax)
            newList = []
            for i in range(0, listLen, skipNum):
                newList.append(tickList[i])
            return newList
        else:
            return tickList

    def getTickTimes(self, strtDt, endDt, td, window):
        # Generate all possible tick times in between start and end times
        tickTimes = [strtDt]
        currDt = strtDt + td
        while currDt <= endDt:
            tickTimes.append(currDt)
            currDt = currDt + td

        # Reduce the number of ticks if there are too many
        if self.tickDiff is None:
            tickTimes = self.limitTicks(tickTimes)

        # Creates lists mapping datetime objs to UTC strings and FFTime ticks/values
        tickStrs = list(map(self.dateTimeToTmstmp, tickTimes))
        tickVals = list(map(self.tickStrToVal, tickStrs))

        # If ticks start at a given offset, subtract it from all tick values
        for i in range(0, len(tickVals)):
            tickVals[i] = tickVals[i] - self.tickOffset

        # Create a list of time value/string tuples to use in setTicks function
        tickPairs = []
        numTicks = len(tickStrs)
        for i in range(0, numTicks):
            # Format strings according to label format
            tickStrs[i] = self.fmtTimeStmp(window, tickStrs[i])
            tickPairs.append((tickVals[i], tickStrs[i]))
        return [tickPairs,[]] # Second list is for 'minor ticks'

    def updateTicks(self, window, mode):
        # In case sliders swapped, just get min/max times
        minTime = min(window.tO, window.tE)
        maxTime = max(window.tO, window.tE)

        # Convert start/end times to strings
        minTimeStr = str(FFTIME(minTime, Epoch=window.epoch).UTC)
        maxTimeStr = str(FFTIME(maxTime, Epoch=window.epoch).UTC)

        # Convert start/end times to datetime objects
        minDt = self.tmstmpToDateTime(minTimeStr)
        maxDt = self.tmstmpToDateTime(maxTimeStr)

        # Identify first and last times that should appear in range
        td = self.modeToDelta[mode] if self.tickDiff is None else self.tickDiff
        strtDt = self.firstTickDt(minDt, td)
        endDt = self.lastTickDt(maxDt, td)

        # Get all tick labels/values in-between and update on plot
        tickPairs = self.getTickTimes(strtDt, endDt, td, window)
        self.setTicks(tickPairs)

    def setCstmTickSpacing(self, diff):
        self.tickDiff = diff
        self.update()

    def resetTickSpacing(self):
        self.tickDiff = None
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

class SpectraPlotItem(pg.PlotItem):
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

class BLabelItem(pg.LabelItem):
    def setHtml(self, html):
        self.item.setHtml(html)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

# based off class here, except i wanted a linear version (deleted a lot of stuff i wasnt gonna use to save time)
#https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/graphicsItems/GraphicsLayout.py
# ref for qt layout component
#http://doc.qt.io/qt-5/qgraphicslinearlayout.html
__all__ = ['GraphicsLayout']
class LinearGraphicsLayout(pg.GraphicsWidget):
    """
    Used for laying out GraphicsWidgets in a linear fashion
    """

    def __init__(self, orientation=QtCore.Qt.Vertical, parent=None):
        pg.GraphicsWidget.__init__(self, parent)
        self.layout = QtGui.QGraphicsLinearLayout(orientation)
        self.setLayout(self.layout)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        self.items = []

    def addLayout(self, **kargs):
        """
        Create an empty GraphicsLayout and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`GraphicsLayout.__init__ <pyqtgraph.GraphicsLayout.__init__>`
        Returns the created item.
        """
        layout = LinearGraphicsLayout(QtCore.Qt.Horizontal, **kargs)
        self.addItem(layout)
        return layout
        
    def addItem(self, item):
        self.items.append(item)
        self.layout.addItem(item)

    def itemIndex(self, item):
        for i in range(self.layout.count()):
            if self.layout.itemAt(i).graphicsItem() is item:
                return i
        raise Exception("Could not determine index of item " + str(item))

    def removeItem(self, item):
        """Remove *item* from the layout."""
        ind = self.itemIndex(item)
        self.layout.removeAt(ind)
        self.scene().removeItem(item)
        self.items = [x for x in self.items if x != item]
        self.update()

    def clear(self):
        for i in self.items:
            self.removeItem(i)

    def setContentsMargins(self, *args):
        # Wrap calls to layout. This should happen automatically, but there
        # seems to be a Qt bug:
        # http://stackoverflow.com/questions/27092164/margins-in-pyqtgraphs-graphicslayout
        self.layout.setContentsMargins(*args)

    def setSpacing(self, *args):
        self.layout.setSpacing(*args)

# same as pdi but with better down sampling (bds)
class PlotDataItemBDS(pg.PlotCurveItem):
    def __init__(self, *args, **kwargs):
        pg.PlotCurveItem.__init__(self, *args, **kwargs)

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


# this class is exact copy of pg.PlotCurveItem but with a changed paint function to draw points instead of lines
# and i removed some random stuff i dont need as well
# prob could have just override that function only but im an idiot and its too late lol. got rid of a bunch of unnecessary stuff at least
import numpy as np
from pyqtgraph import functions as fn
from pyqtgraph import Point
import struct, sys
from pyqtgraph import getConfigOption
__all__ = ['PlotPointsItem']
class PlotPointsItem(pg.GraphicsObject):
    """
    Class representing a single plot curve. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.
    
    Features:
    
    - Fast data update
    - Fill under curve
    - Mouse interaction
    
    ====================  ===============================================
    **Signals:**
    sigPlotChanged(self)  Emitted when the data being plotted has changed
    sigClicked(self)      Emitted when the curve is clicked
    ====================  ===============================================
    """
    
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)
    
    def __init__(self, *args, **kargs):
        """
        Forwards all arguments to :func:`setData <pyqtgraph.PlotCurveItem.setData>`.
        
        Some extra arguments are accepted as well:
        
        ==============  =======================================================
        **Arguments:**
        parent          The parent GraphicsObject (optional)
        clickable       If True, the item will emit sigClicked when it is 
                        clicked on. Defaults to False.
        ==============  =======================================================
        """
        pg.GraphicsObject.__init__(self, kargs.get('parent', None))
        self.clear()
            
        ## this is disastrous for performance.
        #self.setCacheMode(QtGui.QGraphicsItem.DeviceCoordinateCache)
        
        self.metaData = {}
        self.opts = {
            'pen': fn.mkPen('w'),
            'name': None,
            'connect': 'all',
            'mouseWidth': 8, # width of shape responding to mouse click
            'compositionMode': None,
        }
        self.setClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)
        
    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints
    
    def name(self):
        return self.opts.get('name', None)
    
    def setClickable(self, s, width=None):
        """Sets whether the item responds to mouse clicks.
        
        The *width* argument specifies the width in pixels orthogonal to the
        curve that will respond to a mouse click.
        """
        self.clickable = s
        if width is not None:
            self.opts['mouseWidth'] = width
            self._boundingRect = None        
        
    def setCompositionMode(self, mode):
        """Change the composition mode of the item (see QPainter::CompositionMode
        in the Qt documentation). This is useful when overlaying multiple items.
        
        ============================================  ============================================================
        **Most common arguments:**
        QtGui.QPainter.CompositionMode_SourceOver     Default; image replaces the background if it
                                                      is opaque. Otherwise, it uses the alpha channel to blend
                                                      the image with the background.
        QtGui.QPainter.CompositionMode_Overlay        The image color is mixed with the background color to 
                                                      reflect the lightness or darkness of the background.
        QtGui.QPainter.CompositionMode_Plus           Both the alpha and color of the image and background pixels 
                                                      are added together.
        QtGui.QPainter.CompositionMode_Multiply       The output is the image color multiplied by the background.
        ============================================  ============================================================
        """
        self.opts['compositionMode'] = mode
        self.update()
        
    def getData(self):
        return self.xData, self.yData
        
    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        ## Need this to run as fast as possible.
        ## check cache first:
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]
        
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (None, None)
            
        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x

        ## If an orthogonal range is specified, mask the data now
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]
            #d2 = d2[mask]
            
        if len(d) == 0:
            return (None, None)

        ## Get min/max (or percentiles) of the requested data range
        if frac >= 1.0:
            # include complete data range
            # first try faster nanmin/max function, then cut out infs if needed.
            b = (np.nanmin(d), np.nanmax(d))
            if any(np.isinf(b)):
                mask = np.isfinite(d)
                d = d[mask]
                if len(d) == 0:
                    return (None, None)
                b = (d.min(), d.max())
                
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            # include a percentile of data range
            mask = np.isfinite(d)
            d = d[mask]
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])
            
        ## Add pen width only if it is non-cosmetic.
        pen = self.opts['pen']
        if not pen.isCosmetic():
            b = (b[0] - pen.widthF()*0.7072, b[1] + pen.widthF()*0.7072)
            
        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b
            
    def pixelPadding(self):
        pen = self.opts['pen']
        w = 0
        if pen.isCosmetic():
            w += pen.widthF()*0.7072
        if self.clickable:
            w = max(w, self.opts['mouseWidth']//2 + 1)
        return w

    def boundingRect(self):
        if self._boundingRect is None:
            (xmn, xmx) = self.dataBounds(ax=0)
            (ymn, ymx) = self.dataBounds(ax=1)
            if xmn is None or ymn is None:
                return QtCore.QRectF()
            
            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions    
                px, py = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0
                
                # return bounds expanded by pixel size
                px *= pxPad
                py *= pxPad
            #px += self._maxSpotWidth * 0.5
            #py += self._maxSpotWidth * 0.5
            self._boundingRect = QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
            
        return self._boundingRect
    
    def viewTransformChanged(self):
        self.invalidateBounds()
        self.prepareGeometryChange()
        
    def invalidateBounds(self):
        self._boundingRect = None
        self._boundsCache = [None, None]
            
    def setPen(self, *args, **kargs):
        """Set the pen used to draw the curve."""
        self.opts['pen'] = fn.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setData(self, *args, **kargs):
        """
        =============== ========================================================
        **Arguments:**
        x, y            (numpy arrays) Data to show 
        pen             Pen to use when drawing. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        connect         Argument specifying how vertexes should be connected
                        by line segments. Default is "all", indicating full
                        connection. "pairs" causes only even-numbered segments
                        to be drawn. "finite" causes segments to be omitted if
                        they are attached to nan or inf values. For any other
                        connectivity, specify an array of boolean values.
        compositionMode See :func:`setCompositionMode 
                        <pyqtgraph.PlotCurveItem.setCompositionMode>`.
        =============== ========================================================
        
        If non-keyword arguments are used, they will be interpreted as
        setData(y) for a single argument and setData(x, y) for two
        arguments.
        
        
        """
        self.updateData(*args, **kargs)
        
    def updateData(self, *args, **kargs):
        if 'compositionMode' in kargs:
            self.setCompositionMode(kargs['compositionMode'])

        if len(args) == 1:
            kargs['y'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        
        if 'y' not in kargs or kargs['y'] is None:
            kargs['y'] = np.array([])
        if 'x' not in kargs or kargs['x'] is None:
            kargs['x'] = np.arange(len(kargs['y']))
            
        for k in ['x', 'y']:
            data = kargs[k]
            if isinstance(data, list):
                data = np.array(data)
                kargs[k] = data
            if not isinstance(data, np.ndarray) or data.ndim > 1:
                raise Exception("Plot data must be 1D ndarray.")
            if 'complex' in str(data.dtype):
                raise Exception("Can not plot complex data types.")

        self.invalidateBounds()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.yData = kargs['y'].view(np.ndarray)
        self.xData = kargs['x'].view(np.ndarray)

        # can def make this faster
        # not sure what format qpolygonf needs tho (saving this progress)
        #https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/functions.py @ line 1470
        # can turn x,y data into QDataStream 
        # then do stream >> poly and just read it all into the polygon insta
        #http://doc.qt.io/qt-5/qpolygonf.html#operator-gt-gt
        # prob way faster, thats what they do except with qpath thing instead!!
        # think first thing written to stream needs to be length, then each point x,y

        #n = self.xData.shape[0]
        #arr = np.empty(n+1, dtype=[('x','>f8'), ('y','>f8')])
        #bv = arr.view(dtype=np.ubyte)
        #arr[:]['x'] = self.xData
        #arr[:]['y'] = self.yData

        #buf = QtCore.QByteArray.fromRawData(arr.data)
        #ds = QtCore.QDataStream()
        #ds >> self.poly
        
        #lendata = len(self.xData)
        ##ds.writeInt(lendata)
        #for i in range(lendata):
        #    ds.writeFloat(self.xData[i])
        #    ds.writeFloat(self.yData[i])

        points = [QtCore.QPointF(x,y) for x,y in zip(self.xData,self.yData)]
        self.poly = QtGui.QPolygonF(points)
        
        
        if 'name' in kargs:
            self.opts['name'] = kargs['name']
        if 'connect' in kargs:
            self.opts['connect'] = kargs['connect']
        if 'pen' in kargs:
            self.setPen(kargs['pen'])
  
        self.update()

        self.sigPlotChanged.emit(self)

    # make overload of plotcurveitem
    # TODO override this method and dont do the getpath crap instead try
    # using qpainter and its drawpoints function???
    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return
        
        p.setRenderHint(p.Antialiasing, False) # just want PIXELS BOY
        
        cmode = self.opts['compositionMode']
        if cmode is not None:
            p.setCompositionMode(cmode)
            
        cp = fn.mkPen(self.opts['pen']) 
        p.setPen(cp)

        p.drawPoints(self.poly)

        
    def clear(self):
        self.xData = None  ## raw values
        self.yData = None
        self.xDisp = None  ## display values (after log / fft)
        self.yDisp = None
        self._boundsCache = [None, None]
        #del self.xData, self.yData, self.xDisp, self.yDisp, self.path

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
    self.png = fn.makeQImage(bg, alpha=True)
    
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
    if nodes is None:  ## nodes maps all node IDs to their XML element. 
                       ## this allows us to ensure all elements receive unique names.
        nodes = {}
        
    if root is None:
        root = item
                
    ## Skip hidden items
    if hasattr(item, 'isVisible') and not item.isVisible():
        return None
        
    ## If this item defines its own SVG generator, use that.
    if hasattr(item, 'generateSvg'):
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
                coords = fn.transformCoordinates(tr, coords, transpose=True)
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
                    nc = fn.transformCoordinates(tr, np.array([[float(x),float(y)]]), transpose=True)
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
                s = fn.transformCoordinates(tr, np.array([[w,0], [0,0]]), transpose=True)
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