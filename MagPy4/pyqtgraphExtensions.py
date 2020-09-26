
from PyQt5 import QtGui, QtCore, QtWidgets
from .dataDisplay import UTCQDate
import pyqtgraph as pg
import pyqtgraph.exporters
from datetime import datetime, timedelta
from .timeManager import TimeManager
from fflib import ff_time
from math import ceil
import functools
from bisect import bisect_left, bisect_right

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
        self.labelHidden = False

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
    
    def hideLabel(self):
        self.regionItems[self.labelPltIndex].lines[0].setVisible(False)

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
        t0 = ff_time.tick_to_date(x0, self.window.epoch)
        t1 = ff_time.tick_to_date(x1, self.window.epoch)

        timeEdit.setStartNoCallback(min(t0,t1))
        timeEdit.setEndNoCallback(max(t0,t1))

    def setMovable(self, val):
        for item in self.regionItems:
            item.setMovable(val)

    def mouseClickEvent(self, ev):
        # Signal this region to be set as the active region
        self.onRegionActivated()

class TrackerRegion(LinkedRegion):
    def __init__(self, *args, **kwargs):
        LinkedRegion.__init__(self, *args, **kwargs)
        for region in self.regionItems:
            for line in region.lines:
                pen = line.pen
                pen.setStyle(QtCore.Qt.DashLine)
                line.setPen(pen)

                pen = line.hoverPen
                pen.setWidth(1)
                pen.setStyle(QtCore.Qt.DashLine)
                line.setHoverPen(pen)

            brush = pg.mkBrush(0, 0, 0, 0)
            region.setHoverBrush(brush)
            region.setBrush(brush)
            region.lines[0].setVisible(False)

    def mouseClickEvent(self, ev):
        return

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
        hoverBrush = pg.mkColor(color)
        hoverBrush.setAlpha(50)
        self.setHoverBrush(pg.mkBrush(hoverBrush))

        # Set line's pen to same color but opaque and its hover pen to black
        for line in self.lines:
            pen = pg.mkPen('#000000')
            pen.setWidth(1)
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

# subclass based off example here:
# https://github.com/ibressler/pyqtgraph/blob/master/examples/customPlot.py

class GridGraphicsLayout(pg.GraphicsLayout):
    def __init__(self, window=None, *args, **kwargs):
        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.window = window
        self.lastWidth = 0
        self.lastHeight = 0

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        """
        Add an item to the layout and place it in the next available cell (or in the cell specified).
        The item must be an instance of a QGraphicsWidget subclass.
        """
        if row is None:
            row = self.currentRow
        if col is None:
            col = self.currentCol
            
        self.items[item] = []
        for i in range(rowspan):
            for j in range(colspan):
                row2 = row + i
                col2 = col + j
                if row2 not in self.rows:
                    self.rows[row2] = {}
                self.rows[row2][col2] = item
                self.items[item].append((row2, col2))

        borderRect = QtGui.QGraphicsRectItem()

        borderRect.setParentItem(self)
        borderRect.setZValue(1e3)
        borderRect.setPen(pg.mkPen(self.border))

        self.itemBorders[item] = borderRect

        item.geometryChanged.connect(self._updateItemBorder)

        self.layout.addItem(item, row, col, rowspan, colspan)
                               # Allows some PyQtGraph features to also work without Qt event loop.
        
        self.nextColumn()

from .plotBase import MagPyPlotItem

class SpectraPlotItem(MagPyPlotItem):
    # plotItem subclass so we can set Spectra plots to be square
    def __init__(self, window=None, *args, **kargs):
        self.squarePlot = False
        MagPyPlotItem.__init__(self, *args, **kargs)

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
        ''' Returns list of items in each row as a list '''
        # Get list of row numbers in sorted order
        rowIndices = list(self.rows.keys())
        rowIndices.sort()

        # Get list of items in each row
        rowItems = []
        for row in rowIndices:
            rowItems.append(self.getRow(row))

        return rowItems

class SpectraGrid(RowGridLayout):
    def addItem(self, *args, **kwargs):
        super().addItem(*args, **kwargs)
        self.updateRowWidths()
    
    def updateRowWidths(self):
        rowItems = self.getRowItems()
        for row in rowItems:
            rowPlots = []
            minWidth = 10
            for item in row:
                if isinstance(item, pg.PlotItem):
                    rowPlots.append(item)
                    width = item.minimumWidth()
                    minWidth = max(width, minWidth)

            for plot in rowPlots:
                plot.setMinimumWidth(minWidth)

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
            {'name':'Orientation: ', 'type':'list', 
                'values': ['Portrait', 'Landscape']
            },
            {'name':'Aspect Ratio: ', 'type':'list', 
                'values':['Original', '4x6', '5x7', '8x10']
            }
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

        # Enable clipping for main plot grid if its in the scene
        mainGrid = None
        sceneItems = self.getScene().items()
        from .MagPy4UI import MainPlotGrid
        for si in sceneItems:
            if isinstance(si, MainPlotGrid):
                mainGrid = si
                mainGrid.enablePDFClipping(True)

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
        oldSource = self.getSourceRect()

        # Get aspect ratio and apply it to image
        aspect = self.params['Aspect Ratio: ']
        widget = None
        if aspect != 'Original':
            # Map the aspect ratio to width and height in inches
            hzAspect, vtAspect = [float(v) for v in aspect.split('x')]
            if horzLt: # Flip if horizontal layout
                hzAspect, vtAspect = vtAspect, hzAspect

            # Get the central widget for the view
            widget = self.getCentralWidget()
            oldSource = widget.boundingRect()

            # Resize the layout to given scale ratio
            scaledRect = self.getScaledRect(hzAspect, vtAspect)
            widget.resize(scaledRect.width(), scaledRect.height())

            # Get new source rect from layout's bounding rect
            sourceRect = widget.boundingRect()
            tr = widget.viewTransform()
            sourceRect = tr.mapRect(sourceRect)

            # Item resizing + color plot preparations
            for item in self.getScene().items():
                if hasattr(item, 'resizeEvent'):
                    item.resizeEvent(None)

            for item in self.getScene().items():
                if hasattr(item, 'prepareForExport'):
                    item.prepareForExport()

        else:
            # Default source rect if no aspect ratio is applied
            sourceRect = oldSource

        # Get view and resize according to aspect ratio
        self.getScene().update()

        # Start painter and render scene
        painter = QtGui.QPainter(self.pdfFile)
        try:
            self.setExportMode(True, {'painter': painter})
            self.getScene().render(painter, targetRect, QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()

        # Disable clipping for main plot grid if in scene
        if mainGrid:
            mainGrid.enablePDFClipping(False)

        # Return view/widget to original size
        if widget:
            widget.resize(oldSource.width(), oldSource.height())

        # Reset color plots
        for item in self.getScene().items():
            if hasattr(item, 'resetAfterExport'):
                item.resetAfterExport()

    def getScaledRect(self, width, height):
        # Returns a rect w/ width = width in inches, height = height in inches
        # in pixel coordinates
        hzDpi = QtGui.QDesktopWidget().logicalDpiX()
        vtDpi = QtGui.QDesktopWidget().logicalDpiY()

        return QtCore.QRectF(0, 0, width*hzDpi, height*vtDpi)

    def getCentralWidget(self):
        view = self.getScene().getViewWidget()
        widget = view.centralWidget
        return widget

# Add PDF exporter to list of exporters
PDFExporter.register()