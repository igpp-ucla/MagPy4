
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from FF_Time import FFTIME

# custom extensions to pyqtgraph for the projects needs

# mostly just slightly edited versions of originals
# vertical by default
# also with built in label support (infiniteline has this but doesnt how i want)
class LinkedInfiniteLine(pg.InfiniteLine):
    def __init__(self, callback, mylabel=None, labelColor=None, *args, **kwds):
        pg.InfiniteLine.__init__(self, *args, **kwds)
        self.callback = callback

        if mylabel and labelColor:
            opts = {'movable':False, 'position':1.0, 'color':labelColor }
            self.mylabel = LinkedInfLineLabel(self, text=mylabel, **opts)

    def mouseDragEvent(self, ev):
        pg.InfiniteLine.mouseDragEvent(self, ev)
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            self.callback(self.getXPos())

class LinkedInfLineLabel(pg.InfLineLabel):
    #def __init__(self, *args, **kwds):
    #    pg.InfLineLabel.__init__(self, *args,**kwds)

    def updatePosition(self):
        # update text position to relative view location along line
        self._endpoints = (None, None)
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return
        pt = pt2 * self.orthoPos + pt1 * (1-self.orthoPos)
        self.setPos(pt)
        self.setAnchor((0.5,0.0)) # in middle of line

    def valueChanged(self): # default behaviour will override mylabel changes
        pass

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont = QtGui.QFont()
        self.tickFont.setPixelSize(14)
        self.style['maxTextLevel'] = 1 # never have any subtick labels
        self.style['textFillLimits'] = [(0,1.1)] # try to always draw labels
        #self.style['tickLength'] = -10
        #todo: override AxisItem generateDrawSpecs and custom set tick length

    def tickStrings(self, values, scale, spacing):
        return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below

    def tickSpacing(self, minVal, maxVal, size):
        #levels = pg.AxisItem.tickSpacing(self,minVal,maxVal,size)
        levels = [(10.0,0),(1.0,0),(0.5,0)]
        return levels

    # overriden from source to be able to have superscript text
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
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


# subclass based off example here:
# https://github.com/ibressler/pyqtgraph/blob/master/examples/customPlot.py
class DateAxis(pg.AxisItem):
    def toUTC(x, window, showFull=False): # converts seconds since epoch to UTC string
        # get current selected range, if greater than hour then show hour marks
        rng = window.getSelectedTimeRange()
        t = str(FFTIME(x, Epoch=window.epoch).UTC)
        t = t.split(' ')[-1]
        if showFull:
            return t

        #t format at this point hh:mm:ss.mmm
        if rng > 30*60: # if over half hour show hh:mm:ss
            return t.rsplit('.',1)[0]
        elif rng > 5: # if over 5 seconds show mm:ss
            return t.split(':',1)[1].split('.')[0]
        return t.split(':',1)[1] # else show mm:ss.mmm

    def tickStrings(self, values, scale, spacing):
        return [DateAxis.toUTC(x,self.window) for x in values]

class GridGraphicsLayout(pg.GraphicsLayout):
    def clear(self):  # clear doesnt get rid of grid layout formatting correctly, todo: make override of this
        pg.GraphicsLayout.clear(self)
        self.currentRow = 0
        self.currentCol = 0

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
    

# this class is exact copy of pg.PlotCurveItem but with a changed paint function to draw points instead of lines
# and i removed some random stuff i dont need as well
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

            