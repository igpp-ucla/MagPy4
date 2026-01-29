
from PyQt5 import QtGui, QtCore, QtWidgets, QtPrintSupport
import pyqtgraph as pg
from fflib import ff_time
import functools
import os
import tempfile
import subprocess

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
        self.color = color

        pg.LinearRegionItem.__init__(self, values=(0,0))

        for plt in self.plotItems:
            self.add_plot(plt, values)

        # Initialize region label at top-most plot
        self.labelPltIndex = 0 if lblPos == 'top' else len(self.plotItems) - 1
        self.setLabel(self.labelPltIndex)
        if self.linkedTE:
            self.updateTimeEditByLines(self.linkedTE)

    def add_plot(self, plot, values=None):
        # Create a LinearRegionItem for each plot with same initialized vals
        values = self.getRegion() if values is None else values
        regionItem = LinkedSubRegion(self, values=values, color=self.color)
        regionItem.setBounds([self.window.minTime, self.window.maxTime])
        plot.addItem(regionItem)
        self.regionItems.append(regionItem)

        # Connect plot's regions/lines to each other
        line0, line1 = regionItem.lines
        line0.sigDragged.connect(functools.partial(self.linesChanged, line0, 0))
        line1.sigDragged.connect(functools.partial(self.linesChanged, line1, 1))
    
        return regionItem

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
        self.regionItems[self.labelPltIndex].lines[0].label.setVisible(False)

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
            self.set_region_style(region)
        self.hideLabel()

    def set_region_style(self, region):
        # Sets dashed line style and no hover filler brush
        for line in region.lines:
            pen = line.pen
            pen.setStyle(QtCore.Qt.DashLine)
            line.setPen(pen)

            pen = line.hoverPen
            pen.setWidth(1)
            pen.setStyle(QtCore.Qt.DashLine)
            line.setHoverPen(pen)
        line.setVisible(False)

        brush = pg.mkBrush(0, 0, 0, 0)
        region.setHoverBrush(brush)
        region.setBrush(brush)
        region.setBounds([self.window.minTime, self.window.maxTime])

    def mouseClickEvent(self, ev):
        return
    
    def add_plot(self, plot, values=None):
        # Adds a new plot region item to plot
        item = super().add_plot(plot, values=values)
        self.set_region_style(item)

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

class BLabelItem(pg.LabelItem):
    def setHtml(self, html):
        self.item.setHtml(html)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()

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

# pg.ViewBoxMenu.ViewBoxMenu.updateState = vbMenu_UpdateState

from pyqtgraph.parametertree import Parameter
import pyqtgraph.exporters
class PDFExporter(pyqtgraph.exporters.Exporter):
    Name = "PDF Document"
    allowCopy = False

    def __init__(self, item):
        pyqtgraph.exporters.Exporter.__init__(self, item)

        # Orientation option
        self.params = Parameter(name='params', type='group', children=[
            {'name':'Orientation: ', 'type':'list', 
                'limits': ['Portrait', 'Landscape'], 'value': 'Portrait'
            },
            {'name':'Aspect Ratio: ', 'type':'list', 
                'limits':['Original', '4x6', '5x7', '8x10'], 'value': 'Original'
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
        # from .MagPy4UI import MainPlotGrid
        # for si in sceneItems:
        #     if isinstance(si, MainPlotGrid):
        #         mainGrid = si
        #         mainGrid.enablePDFClipping(True)

        # Set page orientation if user selected 'Landscape' mode
        horzLt = self.params['Orientation: '] == 'Landscape'
        if horzLt:
            self.pdfFile.setPageOrientation(QtGui.QPageLayout.Landscape)

        # Get the device resolution and set resolution for the PDF Writer
        res = QtWidgets.QDesktopWidget().logicalDpiX()
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
                    try:
                        item.resizeEvent(None)
                    except Exception as e:
                        continue

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
        hzDpi = QtWidgets.QDesktopWidget().logicalDpiX()
        vtDpi = QtWidgets.QDesktopWidget().logicalDpiY()

        return QtCore.QRectF(0, 0, width*hzDpi, height*vtDpi)

    def getCentralWidget(self):
        view = self.getScene().getViewWidget()
        widget = view.centralWidget
        return widget

# TO-DO: Look into adding postscript/eps exporter

# Add PDF exporter to list of exporters
PDFExporter.register()