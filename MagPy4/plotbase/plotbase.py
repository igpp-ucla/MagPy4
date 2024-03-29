import functools
import PyQt5
from fflib import ff_time
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
from dateutil import rrule
from dateutil.rrule import rrule
from dateutil.rrule import rrule, SECONDLY, MINUTELY, HOURLY, DAILY, MONTHLY, YEARLY
from ..common import logbase_map, logbase_reverse
from pyqtgraph.graphicsItems.PlotDataItem import PlotDataset

class LogAction(QtWidgets.QWidgetAction):
    toggled = QtCore.pyqtSignal(bool)
    baseChanged = QtCore.pyqtSignal(str)
    def __init__(self, axis='X', parent=None) -> None:
        super().__init__(parent)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        
        # Set up checkbox
        self.checkbox = QtWidgets.QCheckBox(f'Log {axis}')
        self.checkbox.toggled.connect(self.toggled)
        
        # Set up base info dropdown
        label = QtWidgets.QLabel('      Base: ')
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(list(logbase_map.keys()))
        self.dropdown.currentTextChanged.connect(self.baseChanged)

        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(label)
        sublayout.addWidget(self.dropdown)
        sublayout.addStretch()

        # Set widget for this item
        layout.addWidget(self.checkbox, 0, 0, 1, 1)
        layout.addLayout(sublayout, 1, 0, 1, 1)
        self.setDefaultWidget(widget)

class StackedAxisLabel(pg.GraphicsLayout):
    def __init__(self, lbls, angle=90, *args, **kwargs):
        self.lblTxt = lbls
        self.sublabels = []
        self.angle = angle
        pg.GraphicsLayout.__init__(self, *args, **kwargs)

        orientation = QtCore.Qt.Vertical if angle in [0, 270] else QtCore.Qt.Horizontal
        self.orientation = orientation

        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setupLabels(lbls)

    def setHtml(self, *args):
        return

    def getLabelText(self):
        return self.lblTxt

    def setupLabels(self, lbls):
        self.lblTxt = lbls
        self.setupSublayout()
        if self.angle != 0:
            lbls = lbls[::-1]
        self.sublayout.addStretch()
        for i in range(0, len(lbls)):
            item = pg.LabelItem(lbls[i], angle=self.angle)
            self.sublabels.append(item)
            self.sublayout.addItem(item)
        self.sublayout.addStretch()

    def setupSublayout(self):
        if self.layout.rowCount() > 0 and self.layout.columnCount() > 0:
            layout_item = self.layout.itemAt(0, 0)
            if layout_item is not None:
                for item in self.sublabels:
                    self.sublayout.removeItem(item)
                    item.deleteLater()
                self.layout.removeItem(layout_item)

        self.sublabels = []
        self.sublayout = QtWidgets.QGraphicsLinearLayout()
        self.sublayout.setOrientation(self.orientation)
        self.sublayout.setSpacing(0)
        self.layout.addItem(self.sublayout, 0, 0, 1, 1)

    def getFont(self):
        # Returns the font w/ the correct point size set
        if self.sublabels == []:
            return QtGui.QFont()

        font = self.sublabels[0].item.font()
        if 'size' in self.sublabels[0].opts:
            fontSize = self.sublabels[0].opts['size'][:-2]
            fontSize = float(fontSize)
        else:
            fontSize = QtGui.QFont().pointSize()

        fontSize = min(fontSize, 12)
        font.setPointSize(fontSize)
        return font

class FadeAnimation(QtCore.QPropertyAnimation):
    ''' Animation for QGraphicsOffacityEffect 'effect' that changes opacity
        from start/end values in rng
    '''
    def __init__(self, effect, rng=(0.0, 1.0), length=150, *args, **kwargs):
        super().__init__(effect, QtCore.QByteArray(b'opacity'))
        a, b = rng
        self.setDuration(length)
        self.setStartValue(a)
        self.setEndValue(b)

class TimedFadeAnimation(QtCore.QSequentialAnimationGroup):
    ''' Animation that fades object in and out with a pause of length pause_len
        in between animations
    '''
    def __init__(self, target, pause_len=3000, fade_len=150, *args, **kwargs):
        self.target = target
        self.effect = QtWidgets.QGraphicsOpacityEffect()
        self.target.setGraphicsEffect(self.effect)

        # Fade in / out animations
        self.start_anim = FadeAnimation(self.effect, (0.0, 1.0), fade_len)
        self.end_anim = FadeAnimation(self.effect, (1.0, 0.0), fade_len)

        super().__init__(target, *args, **kwargs)

        self.addAnimation(self.start_anim)
        self.addPause(pause_len)
        self.addAnimation(self.end_anim)

class TimedMessage(pg.TextItem):
    ''' Message that fades in / out in center of parent and
        is displayed for ms ms
    '''
    def __init__(self, msg='', ms=1000, *args, **kwargs):
        self.msg = msg
        self.length = ms
        super().__init__(msg, *args, **kwargs)
        self.setAnchor((0.5, 0.5))
        self.setVisible(False)
        self.animation = None

    def updatePosition(self):
        ''' Centers timed message in parent's bounding rect '''
        pos = self.parentItem().boundingRect().center()
        self.setPos(pos)

    def showMessage(self, txt=None):
        ''' Shows the message txt in the center of the screen
            with a fade in/out animation
        '''
        # Set visible on first showing (animation does not work if hidden)
        if self.animation is None:
            self.setVisible(True)

        # Centers message on screen and updates text
        self.updatePosition()
        self.setText(self.msg if txt is None else txt)

        # Create/start fade in/out animation
        if self.animation is None:
            self.animation = TimedFadeAnimation(self, self.length)
            self.animation.effect.setOpacity(1.0)
            QtCore.QTimer.singleShot(self.length, self.animation.end_anim.start)
        else:
            self.fade()

    def fade(self):
        ''' (Re)starts fade animation '''
        self.animation.stop()
        self.animation.start()

class GraphicsLayout(pg.GraphicsLayout):
    def __init__(self, *args, **kwargs):
        # Indicates whether user is able to to toggle tracking
        self.trackingEnabled = None
        self.textFuncs = {}

        # Default init
        super().__init__(*args, **kwargs)

        # Create timed status message object and set attributes
        self.message = TimedMessage(color=(255, 255, 255), fill=(0, 0, 0, 175))
        self.message.setParentItem(self)

        font = QtGui.QFont()
        font.setPointSize(14)
        self.message.setFont(font)
        self.message.setZValue(2000)

    def getItems(self):
        return [item for item in self.items]

    def getPlots(self):
        return [item for item in self.getItems() if isinstance(item, MagPyPlotItem)]

    def enableTracking(self, val, textFuncs={}, viewWidget=None):
        ''' Enables/disables tracking based on val and passes textFuncs
            to each plot's enableTracking function,
            viewWidget specifies the widget that can be used to set the shortcut
        '''
        # Enable/disable tracking for each plot
        self.trackingEnabled = val
        self.textFuncs = textFuncs
        plots = self.getPlots()
        for plt in plots:
            plt.enableTracking(val, textFuncs)

        # Create tracking shortcut
        if val and viewWidget:
            viewWidget.add_shortcut('T', self.toggleTracking)
    
    def toggleTracking(self):
        # Skip if tracking not available
        if not self.trackingEnabled:
            self.message.showMessage('Tracking not enabled for this view')
            return

        # Toggle tracking for each plot and get its current tracking mode
        msg = ''
        plots = self.getPlots()
        for plt in plots:
            plt.toggleTracking()
            msg = plt.getTrackingMode()

        # Show timed message indicating tracking status
        if msg is not None:
            self.message.showMessage(msg)

class MagPyViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        self.scroll_enabled = [True, True]
        super().__init__(*args, **kwargs)
        self.window = None
    
    def enable_yscroll(self, val=True):
        self.scroll_enabled[1] = val
    
    def set_window(self, w):
        self.window = w

    def wheelEvent(self, ev, axis=None):
        self.setMouseEnabled(*self.scroll_enabled)
        result = super().wheelEvent(ev, axis=axis)
        self.setMouseEnabled(*self.mouseEnabled())
        return result

    def mouseClickEvent(self, ev):
        if self.window is None:
            super().mouseClickEvent(ev)
            return

        if ev.button() == QtCore.Qt.LeftButton:
            self.onLeftClick(ev)

        else: # asume right click i guess, not sure about middle mouse button click
            self.onRightClick(ev)

        ev.accept()

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        pos = ev.scenePos()
        rect = self.sceneBoundingRect()

        if not rect.contains(pos):
            return

        # Apply left click to plot grid; Window manages behavior
        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)
        self.window.gridLeftClick(x, self, ctrlPressed)

    def onRightClick(self, ev):
        # Attempt to apply right click to plot grid
        res = self.window.gridRightClick(self)
        if not res: # No selections currently active, use default right click method
            pg.ViewBox.mouseClickEvent(self, ev)

    def mouseDragEvent(self, ev, axis=None):
        if self.window is None:
            pg.ViewBox.mouseDragEvent(self, ev)
            return

        if ev.isFinish(): # on release
            if ev.button() == QtCore.Qt.LeftButton:
                self.onLeftClick(ev)
            elif ev.button() == QtCore.Qt.RightButton:
                self.onRightClick(ev)
        ev.accept()
    
    def get_plot(self):
        parent = self.parentLayoutItem()
        if parent is not None:
            parent = parent.parentLayoutItem()
        return parent

    def hoverEvent(self, ev):
        # Update hover tracker position if present
        self.get_plot().vbHoverEvent(ev)

class MagPyPlotItem(pg.PlotItem):
    sigLogChanged = QtCore.pyqtSignal(tuple)
    def __init__(self, epoch=None, name=None, selectable=False, *args, **kwargs):
        # Initialize axis items
        axisKws = ['left', 'top', 'right', 'bottom']
        axisItems = {kw:MagPyAxisItem(kw) for kw in axisKws}
        vb = MagPyViewBox()

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
        self.logMenu = None

        # Hover tracking disabled by default
        self.hoverTracker = None

        # Log bases empty by default
        self.logbase = [None, None]

        super().__init__(viewBox=vb, axisItems=axisItems, *args, **kwargs)
        for ax in axisItems:
            axisItems[ax].setZValue(1000)
        self.hideButtons()

    def isSpecialPlot(self):
        '''
            Returns whether this is regular line plot or not
        '''
        return (len(self.get_specs()) > 0)
    
    def setName(self, name):
        ''' Sets internal name/label for this plot '''
        self.name = name
    
    def getName(self, name):
        ''' Returns internal name/label for this plot '''
        return self.name

    def updateLogMode(self):
        x, y = self.getLogMode()
        for i in self.items:
            if hasattr(i, 'setLogMode'):
                i.setLogMode(x,y)
        self.getAxis('bottom').setLogMode(x)
        self.getAxis('top').setLogMode(x)
        self.getAxis('left').setLogMode(y)
        self.getAxis('right').setLogMode(y)

        self.sigLogChanged.emit((None, None))
    
    def getLogMode(self):
        x = self.ctrl.logXCheck.isChecked()
        y = self.ctrl.logYCheck.isChecked()
        return (x, y)
    
    def getLogBase(self):
        return self.logbase
    
    def setLogBase(self, x=None, y=None):
        if x is not None:
            for axdir in ['top', 'bottom']:
                self.getAxis(axdir).setLogBase(x)
            self.logbase[0] = x

        if y is not None:
            for axdir in ['left', 'right']:
                self.getAxis(axdir).setLogBase(y)
            self.logbase[1] = y
        
        for i in self.items:
            if hasattr(i, 'setLogBase'):
                i.setLogBase(*self.logbase)

        self.sigLogChanged.emit((None, None))
        self.updateLogMode()

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
    
    def clear_data(self):
        ''' Clears all data items (traces) from plot '''
        pdis = self.listDataItems()
        for pdi in pdis:
            self.removeItem(pdi)
    
    def get_specs(self):
        from ..tools.dynbase import SpectraGridItem
        grids = []
        for item in self.items:
            if isinstance(item, SpectraGridItem):
                grids.append(item)
        return grids

    def clear_specs(self):
        specs = self.get_specs()
        for spec in specs:
            self.removeItem(spec)

    def load_color_plot(self, specData, winFrame=None, showLabel=True):
        ''' Loads spectrogram from specData object '''
        from ..tools.dynbase import SpectraGridItem
        from ..tools.dynbase import SpectraLegend

        # Extract grid info
        y, x = specData.get_bins()

        # Set log y scaling
        self.logYScale = specData.log_y_scale()
        logbase = specData.get_logbase()
        self.setLogMode(x=False, y=self.logYScale)
        self.setLogBase(y=logbase)

        # Extract ranges and scale data logarithmically
        x_range, y_range = specData.get_ranges()
        ylow, yhigh = y_range
        logfunc = logbase_map[logbase]
        if self.logYScale:
            ys = logfunc(ylow)
            ye = logfunc(yhigh)
        else:
            ys, ye = ylow, yhigh
        self.setYRange(ys, ye, 0.0)

        # Set labels
        left_label, grad_label = specData.get_labels()
        self.getAxis('left').setLabel(left_label)

        # Get spectra grid item
        mapped_grid = specData.get_mapped_grid()
        get_color = lambda r : QtGui.QColor(*r)
        n = len(mapped_grid)
        colorGrid = [list(map(get_color, mapped_grid[i])) for i in range(0, n)]
        grid_obj = SpectraGridItem(y, colorGrid, x)
        grid_obj.spec = specData
        grid_obj.setZValue(-1000)
        self.addItem(grid_obj)

        # Create gradient legend
        legend = SpectraLegend(offsets=(0, 0))
        legend.setPlot(self)
        legend.enableMenu(True)
        legend.setRange(specData.get_gradient(), specData.get_value_range())

        # Create gradient legend label
        lgnd_lbl = StackedAxisLabel(grad_label)
        legend.setLabel(lgnd_lbl)
        legend.setLogMode(specData.log_color_scale())
        return (grid_obj, legend, lgnd_lbl)

    def getSpecData(self):
        datas = []
        specs = self.get_specs()
        for spec in specs:
            if spec.spec is not None:
                datas.append(spec.spec)
        return datas

    def getLineInfo(self):
        '''
            Extract dictionaries containing info about
            the plot data item, trace pen, and name
        '''
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
    
    def listAllDataItems(self):
        grids = self.get_specs()
        return grids + self.listDataItems()

    def addItem(self, item, *args, **kargs):
        if hasattr(item, 'setLogBase'):
            item.setLogBase(*self.logbase)

        return super().addItem(item, *args, **kargs)

    def setVarInfo(self, info):
        self.varInfo = info
    
    def _updateLogMenu(self):
        # Block signals
        self.xlogaction.blockSignals(True)
        self.ylogaction.blockSignals(True)

        # Get current log values
        logx, logy = self.getLogMode()
        basex, basey = self.getLogBase()

        # Initialize values
        self.xlogaction.checkbox.setChecked(logx)
        self.ylogaction.checkbox.setChecked(logy)

        self.xlogaction.dropdown.setCurrentText(basex)
        self.ylogaction.dropdown.setCurrentText(basey)

        self.xlogaction.blockSignals(False)
        self.ylogaction.blockSignals(False)

    def getLogMenu(self):
        if self.logMenu:
            self._updateLogMenu()
            return self.logMenu

        # Set up log menu
        logmenu = QtWidgets.QMenu('Log Menu')
        xlogaction = LogAction('X')
        ylogaction = LogAction('Y')
        self.xlogaction = xlogaction
        self.ylogaction = ylogaction

        logmenu.addAction(xlogaction)
        logmenu.addAction(ylogaction)

        # Connect actions to functions
        xlogaction.toggled.connect(self.ctrl.logXCheck.setChecked)
        ylogaction.toggled.connect(self.ctrl.logYCheck.setChecked)

        xfunc = functools.partial(self.setLogBase, y=None)
        yfunc = functools.partial(self.setLogBase, None)
        xlogaction.baseChanged.connect(xfunc)
        ylogaction.baseChanged.connect(yfunc)

        # Initialize values
        self._updateLogMenu()

        self.logMenu = logmenu
        return logmenu

    def getContextMenus(self, ev):
        menus = [self.getLogMenu(), self.getMenu()]
        return self.getLogMenu()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        ''' Calls dataBounds() on each plot data item
            in plot and returns the min/max upper bounds
            for the plot if valid
        '''
        pdis = self.listAllDataItems()
        lower, upper = None, None
        for pdi in pdis:
            bounds = pdi.dataBounds(ax=ax, frac=frac, orthoRange=orthoRange)
            if bounds is None or bounds[0] is None or bounds[1] is None:
                continue

            if lower is None:
                lower, upper = bounds
            else:
                lower = min(lower, bounds[0])
                upper = max(upper, bounds[-1])

        return (lower, upper)
    
    def enableTracking(self, val=True, format_funcs={}):
        if self.hoverTracker:
            self.removeItem(self.hoverTracker)
            self.hoverTracker.deleteLater()
            self.hoverTracker = None

        if val:
            self.hoverTracker = GridTracker(self, format_funcs)
            self.addItem(self.hoverTracker, ignoreBounds=True)
    
    def getTrackingMode(self):
        if self.hoverTracker:
            return self.hoverTracker.get_mode()
        else:
            return self.hoverTracker.modes[0]
        
    def toggleTracking(self):
        if self.hoverTracker:
            self.hoverTracker.toggle()
    
    def vbHoverEvent(self, ev):
        if self.hoverTracker:
            if ev.isEnter():
                self.hoverTracker.setVisible(True)
            elif ev.isExit():
                self.hoverTracker.setVisible(False)
                return

            pos = ev.pos()
            pos = self.getViewBox().mapToView(pos)
            self.hoverTracker.update_position(pos)

    def clear(self, *args ,**kwargs):
        super().clear(*args, **kwargs)
        if self.hoverTracker:
            self.enableTracking(True)

class Spacer(pg.GraphicsWidget):
    def __init__(self, mode='horizontal'):
        super().__init__()
        expanding = QtWidgets.QSizePolicy.GrowFlag
        maximum = QtWidgets.QSizePolicy.ShrinkFlag
        if mode == 'horizontal':
            left = expanding | maximum
            right = QtWidgets.QSizePolicy.Maximum
        else:
            left = QtWidgets.QSizePolicy.Maximum
            right = expanding | maximum
            self.setMinimumHeight(0)
        
        self.setSizePolicy(QtWidgets.QSizePolicy(left, right))
    
class StackedLabel(pg.GraphicsLayout):
    def __init__(self, labels, colors=[], units=[], size=None):
        self.sublayout = None
        super().__init__()
        # self.setBorder(pg.mkPen((0, 0, 255)))

        # Set up sublayout
        self.sublayout = pg.GraphicsLayout()
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.sublayout.setSizePolicy(sp)
        self.sublayout.setSpacing(-10)
        self.sublayout.setContentsMargins(0, 0, 0, 0)
        spacer1 = Spacer('vertical')
        spacer2 = Spacer('vertical')
        for spacer in [spacer1, spacer2]:
            spacer.setMinimumHeight(0)

        self.addItem(spacer1, 0, 0, 1, 1)
        self.addItem(self.sublayout, 1, 0, 1, 1)
        self.addItem(spacer2, 2, 0, 1, 1)

        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)

        # Add label_items
        self.label_items = []
        self.units = []
        self.labels = []
        self.colors = []
        self.text_size = str(QtGui.QFont().pointSize()) if size is None else str(size) + 'pt'
        self.set_sublabels(labels, colors, units)

    def add_sublabel(self, label, color='#FFFFFF', unit=None, readjust=True):
        self.labels.append(label)
        self.colors.append(color)
        self.units.append(unit if unit is not None else '')
        self.readjust_labels()

    def clear_items(self):
        self.sublayout.clear()
        self.label_items = []
        self.units = []
        self.labels = []
        self.colors = []
    
    def set_sublabels(self, labels, colors=[], units=[]):
        self.clear_items()
        self.labels = labels
        self.label_items = []
        self.units = []
        self.colors = []
        for i in range(len(labels)):
            if i >= len(colors):
                color = '#FFFFFF'
            else:
                color = colors[i]
            if i >= len(units):
                unit = ''
            else:
                unit = units[i]

            label = pg.LabelItem(labels[i], color=color, size=self.text_size)
            label.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum))
            label.setMinimumWidth(2000)
            self.sublayout.addItem(label, i, 0)

            self.units.append(unit)
            self.colors.append(color)
            self.label_items.append(label)
        self._add_units()

    def readjust_labels(self):
        self.set_sublabels(self.labels, self.colors, self.units)
    
    def _add_units(self):
        unit_set = list(set(self.units))
        if len(unit_set) == 1 and unit_set[0] != '':
            unit = unit_set[0]
            unit_label = pg.LabelItem(f'[{unit}]', color='#808080', size=self.text_size)
            unit_label.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum))
            self.sublayout.addItem(unit_label, len(self.labels), 0)
    
    def set_colors(self, colors):
        self.colors = colors
        self.readjust_labels()
    
    def set_labels(self, labels):
        self.labels = labels
        self.readjust_labels()
    
    def get_colors(self):
        return self.colors

    def get_labels(self):
        return self.labels
    
    def set_font_size(self, pt_size, replot=True):
        self.text_size = str(pt_size)+'pt'
        if replot:
            self.readjust_labels()

class MagPyAxisItem(pg.AxisItem):
    ticksChanged = QtCore.pyqtSignal(object)
    textSizeChanged = QtCore.pyqtSignal(object)
    axisClicked = QtCore.pyqtSignal()
    def __init__(self, orientation, pen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True):
        self.tickDiff = None
        self.minWidthSet = False
        self.levels = None
        self.ignore_bounds = False
        self.logbase = '10'
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

    def tickValues(self, values, scale, spacing):
        prev = None
        if self.logMode and self.logbase != '10':
            prev = self.logMode
            self.logMode = False
        result = super().tickValues(values, scale, spacing)
        if prev is not None:
            self.logMode = prev
        return result

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
    
    def setLogBase(self, b):
        if b not in logbase_map:
            raise Exception('Error: Invalid log base given to axis item')
        self.logbase = b
        self.update()

    def logTickStrings(self, values, scale, spacing):
        return [f'{self.logbase}<sup>{int(x)}</sup>' for x in np.array(values).astype(float)]

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
                if s is None and (not self.ignore_bounds):
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, str(s))
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
                finished = finished and (not self.ignore_bounds)
                if finished:
                    break

            # Determine exactly where tick text should be drawn
            visibleRow = []
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                vstr = str(vstr)
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

    def mouseClickEvent(self, ev):
        super().mouseClickEvent(ev)
        if (ev.button() == QtCore.Qt.LeftButton):
            self.axisClicked.emit()

class DateAxis(pg.AxisItem):
    ticksChanged = QtCore.pyqtSignal(object)
    axisClicked = QtCore.pyqtSignal()
    formatChanged = QtCore.pyqtSignal(str)
    labelChanged = QtCore.pyqtSignal(str)
    def __init__(self, epoch, orientation, offset=0, *args, **kwargs):
        self.tickOffset = offset
        self.timeRange = None
        self.epoch = epoch
        self.levels = None
        self.label_visible = False # Keeps track of whether label is visible

        self.tO = 0
        self.tE = 0

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
        self.tO = mn + self.tickOffset
        self.tE = mx + self.tickOffset

        if self.label_visible:
            self.setLabel(self.get_label())

    def get_label_modes(self):
        ''' Returns the full list of label formats available '''
        return list(self.fmt_strs.keys())

    def get_range(self):
        ''' Returns the range of the axis item '''
        return (self.tO, self.tE)

    def get_default_mode(self):
        ''' Returns the time mode set for the axis item '''
        if self.curr_mode is None:
            # If no current time mode is set, get the time
            # range delta and guess mode
            minVal, maxVal = self.get_range()
            start = ff_time.tick_to_date(minVal, self.epoch)
            end = ff_time.tick_to_date(maxVal, self.epoch)
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

    def setLabel(self, *args, **kwargs):
        super().setLabel(*args, **kwargs)
        self.labelChanged.emit(self.get_label())

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
        start = ff_time.tick_to_date(minVal + self.tickOffset, self.epoch)
        end = ff_time.tick_to_date(maxVal + self.tickOffset, self.epoch)
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
            ticks = [ff_time.date_to_tick(date, self.epoch) for date in dates]
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
            date = ff_time.tick_to_date(value, self.epoch)
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
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, str(s))
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
                vstr = str(vstr)
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
    
    def get_levels(self):
        return self.levels

class MagPyPlotDataItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self.connectSegments = None
        super().__init__(*args, **kwargs)
        self.opts['logBase'] = [None, None]

        if isinstance(self.opts['connect'], (list, np.ndarray)):
            self.connectSegments = self.opts['connect']
        elif self.opts['connect'] == 'auto':
            self.opts['connect'] = 'all'
        
    def setLogBase(self, x=None, y=None):
        if x is not None:
            self.opts['logBase'][0] = x

        if y is not None:
            self.opts['logBase'][1] = y
        
        self._clearDisplay()
        self.updateItems()
        self.update()

    def _clearDisplay(self):
        self.xDisp = None
        self.yDisp = None

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
    
    def _get_log_func(self, index):
        logbase = self.opts['logBase'][index]
        if logbase is None:
            logfunc = np.log10
        else:
            logfunc = logbase_map[logbase]
        return logfunc

    def getDisplayDataset(self):
        if self._dataset is None:
            return (None, None)

        if self._datasetDisplay is None:
            x, y = self._dataset.x, self._dataset.y
            mask = None

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
                logfunc = self._get_log_func(0)
                x = logfunc(x)

            if self.opts['logMode'][1]:
                logfunc = self._get_log_func(1)
                mask = y > 0
                y = logfunc(y[mask])
                x = x[mask]

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
                        width *= 1.5 # Prefer smaller downsampling factors
                        ds = int(max(1, int((x1-x0) / (width*self.opts['autoDownsampleFactor']))))

            if self.opts['clipToView']:
                view = self.getViewBox()
                if view is None or not view.autoRangeEnabled()[0]:
                    # this option presumes that x-values have uniform spacing
                    range = view.viewRange()[0]
                    if range is not None and len(x) > 1:
                        lft, rght = range[0], range[-1]
                        x0 = bisect_left(x, lft)
                        x0 = max(0, x0-1)
                        x1 = bisect_right(x, rght+1)
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

            if self.connectSegments is not None:
                self.opts['connect'] = segs

            self._datasetDisplay = PlotDataset(x, y)
        return self._datasetDisplay

    def updateItems(self, *args, **kwargs):
        data = self.getDisplayDataset()
        x, y = data.x, data.y
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
    
class TraceLegend(pg.LegendItem):
    ''' Legend that displays the x,y values for each trace passed to it '''
    def __init__(self, vb, *args, **kwargs):
        self.vb = vb
        super().__init__(*args, **kwargs)
        self.setParentItem(vb)
        self.anchor((1,0), (1,0))

    def paint(self, p, *args, **kwargs):
        # Skip painting if items list is empty
        if len(self.items) == 0:
            return

        # Get rect and remove 5 pixels from each edge
        margin = 5
        rect = self.boundingRect()
        rect = rect.adjusted(margin, margin, -margin, -margin)
        rect = rect.toRect()

        # Draw a slightly rounded rect with grey bg and slight transparency
        pen = pg.mkPen((150, 150, 150, 150))
        brush = pg.mkBrush((240, 240, 240, 200))

        p.setPen(pen)
        p.setBrush(brush)
        p.drawRoundedRect(rect, 2, 2)

    def updateItems(self, items):
        ''' Update items in the legend '''
        # Clear previous items and add each item and its label to legend
        self.clear()
        for pt in items:
            self.addItem(pt, pt.get_labels()[0])

class AnchoredPlotText(pg.TextItem):
    ''' Text anchored to bottom right corner of plot '''
    def __init__(self, plot, anchor=None, *args, **kwargs):
        self.plot = plot
        super().__init__(*args, **kwargs)

        # Anchor to plot's bottom right corner
        self.setParentItem(plot)
        vb = self.plot.getViewBox()
        vb.sigResized.connect(self.update_position)
        self.setAnchor((1, 1))

    def update_position(self):
        ''' Adjust text to bottom right corner of plot '''
        vb = self.plot.getViewBox()
        rect = vb.geometry()
        self.setPos(rect.bottomRight())

class LabeledScatter(pg.ScatterPlotItem):
    ''' Scatter plot item that is able to generate
        labels for each point
    '''
    def __init__(self, format_funcs={}, *args, z_value=None, **kwargs):
        self.format_funcs = format_funcs
        self.labels = []
        self.z_value = z_value
        super().__init__(*args, **kwargs)

    def get_labels(self):
        ''' Get label text for each point '''
        # Get the string label for each point
        labels = []
        pts = self.points()
        for pt in pts:
            # Map position to values based on format_funcs
            pos = pt.pos()
            x, y = pos.x(), pos.y()
            x = str(np.round(x, decimals=5)) if 'x' not in self.format_funcs else self.format_funcs['x'](x)
            y = str(np.round(y, decimals=5)) if 'y' not in self.format_funcs else self.format_funcs['y'](y)
            items = [x, y]

            # Map z value if given
            if self.z_value is not None:
                z = str(self.z_value)
                z = str(z) if 'z' not in self.format_funcs else self.format_funcs['z'](z)
                items.append(z)

            # Save string pair
            label = ', '.join(items)
            label = f'({label})'
            labels.append(label)
        
        return labels
    
    def mouseClickEvent(self, ev):
        # Ignore mouse clicks since this is a cosmetic object only
        ev.ignore()

class CosmeticInfLine(pg.InfiniteLine):
    ''' Infinite line that is not movable by the user and does not accept
        click events
    '''
    def mouseClickEvent(self, ev):
        ev.ignore()

class GridTracker(pg.GraphicsObject):
    ''' Displays information about points that are being hovered over
        or alternatively, displays a set of intersecting lines at
        the mouse position and info about the current position
    '''
    modes = {
        0 : 'Tracking Off',
        1 : 'Tracking Enabled - Points',
        2 : 'Tracking Enabled - Grid'
    }
    def __init__(self, plot, format_funcs={}, pen=None, *args, **kwargs):
        self.plot = plot
        self.mode = 1 # 1 = points mode, 2 = tracker mode, 0 = off
        self._show_spec_vals = True

        # Store formatting functions
        self.format_funcs = format_funcs

        # Set up potential scatter plots to show
        self.hover_pts = []
        self.legend = None
        super().__init__(*args, **kwargs)

        # Set up both lines and label item
        self.vert_line = CosmeticInfLine()
        self.horz_line = CosmeticInfLine(angle=0)
        self.label = AnchoredPlotText(plot, fill=(250, 250, 250, 150))

        # Set pens and colors for items
        if pen is None:
            pen = pg.mkPen((0, 0, 0, 255))
            pen.setStyle(QtCore.Qt.DotLine)

        # Set pen and label colors
        self.vert_line.setPen(pen)
        self.horz_line.setPen(pen)
        self.label.setColor(pg.mkColor(0, 0, 0))

        # Set a high-zalue and parent item for each object
        for obj in [self.vert_line, self.horz_line]:
            obj.setParentItem(self)

        # Hide lines and set z-value        
        self.setZValue(1000)
        self.showLines(False)
    
    def toggle(self):
        ''' Toggles between tracker modes where
            0 = off
            1 = points
            2 = tracker lines
        '''
        self.mode = (self.mode + 1) % 3
        self.setVisible(self.mode != 0)

        # Clear points and hide lines if necessary
        self.clearPoints()
        if self.mode != 2:
            self.showLines(False)

    def enable_spec_values(self, val=True):
        self._show_spec_vals = val

    def setVisible(self, val):
        super().setVisible(val)
        self.label.setVisible(val)
        self.label.setText('')
        if self.legend:
            self.legend.setVisible(val)
            self.legend.clear()

    def boundingRect(self):
        # Get vertical line height
        vert = self.vert_line.boundingRect()
        top = vert.right()
        height = vert.width()

        # Get horizontal line width
        horz = self.horz_line.boundingRect()
        left = horz.left()
        width = horz.width()
        
        rect = QtCore.QRectF(horz.center().x(), vert.center().x(), 1, 1)
        return rect

    def _check_in_vb(self, pos):
        ''' Checks if given position is in viewbox range '''
        x, y = pos.x(), pos.y()
        (xmin, xmax), (ymin, ymax) = self.get_vb().viewRange()
        if x < xmin or x > xmax:
            return False
        elif y < ymin or y > ymax:
            return False
        else:
            return True

    def get_vb(self):
        ''' Returns the viewbox this tracker is in '''
        vb = self.getViewBox()
        if vb is None:
            return self.parentItem()
        return vb
    
    def clearPoints(self):
        ''' Delete and clear old hover items '''
        for item in self.hover_pts:
            item.setParentItem(None)
            item.deleteLater()

        self.hover_pts = []
        if self.legend:
            self.legend.clear()

    def showLines(self, val):
        ''' Shows/hides tracker lines '''
        self.vert_line.setVisible(val)
        self.horz_line.setVisible(val)
        self.label.setVisible(val)

    def update_points(self, pos):
        ''' Update labeled trace points '''
        # Extract scene mouse position and map to view coordinates
        view = self.get_vb()
        x, y = pos.x(), pos.y()

        # Delete old hover items
        self.clearPoints()

        # Get list of child items hovering over
        self.hover_pts = self.get_hover_points(pos)

        # Get list of spec values hovered over
        if self._show_spec_vals:
            self.hover_pts += self.get_hover_squares(pos)

        # Create new legend item if missing
        if self.legend is None:
            self.legend = TraceLegend(self.getViewBox())

        # Update legend items
        self.legend.updateItems(self.hover_pts)
    
    def update_tracker(self, pos):
        ''' Update tracker lines and indicator '''
        # Extract scene mouse position and map to view coordinates
        x, y = pos.x(), pos.y()

        # Hide or show tracker lines if needed
        vis = self.vert_line.isVisible()
        in_rng = self._check_in_vb(pos)
        if vis and not in_rng:
            self.showLines(False)
        elif not vis and in_rng:
            self.showLines(True)

        # Update line positions
        self.horz_line.setPos(y)
        self.vert_line.setPos(x)

        # Update label text
        txt = self.format_text(x, y)
        self.label.setText(txt)

    def update_position(self, pos):
        # Do not update if tracking is off
        if self.mode == 0:
            return
        
        # Otherwise update points or grid tracker
        if self.mode == 1:
            if not self._check_in_vb(pos):
                return
            self.update_points(pos)
        else:
            self.update_tracker(pos)

    def paint(self, p, *args):
        return

    def get_mode(self):
        ''' Get tracking mode (in string format) '''
        txt = GridTracker.modes.get(self.mode)
        return txt

    def format_text(self, x, y):
        ''' Formats x,y values to display in label '''
        # Format x value
        if 'x' in self.format_funcs:
            x = self.format_funcs['x'](x)
        else:
            x = np.format_float_positional(x, precision=5)
        
        # Format y value
        if 'y' in self.format_funcs:
            y = self.format_funcs['y'](y)
        else:
            y = np.format_float_positional(y, precision=5)
        
        # Assemble label text
        txt = f'({x}, {y})'

        return txt

    def get_radius_rect(self, pos):
        ''' Get the rect centered at the cursor position with
            a radius of self.radius
        '''
        # Extract center position info
        x = pos.x()
        y = pos.y()

        # Get the factors to scale viewbox coordinates
        # to pixels by
        xpix, ypix = self.get_vb().viewPixelSize()

        # Create the radius rect (scaled by the pixel length)
        click_radius = 20
        xrad = xpix*click_radius
        yrad = ypix*click_radius
        radius_rect = QtCore.QRectF(x-xrad, y-yrad, xrad*2, yrad*2)
        return radius_rect
    
    def get_plot_traces(self):
        ''' Returns the list of PlotCurveItems in the plot '''
        pdis = self.plot.listDataItems()
        traces = [t.curve for t in pdis if isinstance(t, pg.PlotDataItem)]
        return traces
    
    def get_hover_points(self, pos):
        ''' 
            Creates a list of hover items for each trace that is close
            to the cursor
        '''
        # Get the traces in the plot
        traces = self.get_plot_traces()

        # Get the radius rect centered around pos
        radius_rect = self.get_radius_rect(pos)

        # Find all the traces that intersect with the rect
        matches = []
        for t in traces:
            path = t.mouseShape()
            if path.intersects(radius_rect):
                matches.append(t)
        
        # Create scatter plot items to display at hover point
        if len(matches) > 0:
            hover_items = self.create_hover_points(matches, radius_rect)
        else:
            hover_items = []

        return hover_items
    
    def get_hover_squares(self, pos):
        spec_objects = self.plot.get_specs()
        spec_points = []
        for spec in spec_objects:
            mapped_grid = spec.cached_mapped_grid()
            if mapped_grid is None:
                continue
            x, y, z = spec.spec.values()
            xs, ys = pos.x(), pos.y()
            if self.plot.getAxis('left').logMode: # TODO: Update this for log mode !
                logbase = self.plot.getLogBase()
                ys = logbase_reverse[logbase[1]](ys)
            x_index = bisect_left(x, xs) - 1
            if len(np.array(y).shape) > 1:
                y = y[x_index]
            y_index = bisect_left(y, ys) - 1
            if x_index < len(z[0]) and y_index < len(z) and x_index >= 0 and y_index >= 0:                
                value = z[y_index][x_index]
                mapped_value = mapped_grid[y_index][x_index]
                pen = pg.mkPen(mapped_value)
                brush = pg.mkBrush(mapped_value)
                scatter = LabeledScatter(self.format_funcs, [xs], [ys], 
                    size=10, pen=pen, brush=brush, z_value=value)
                spec_points.append(scatter)
        return spec_points

    def create_hover_points(self, pdis, radius_rect):
        '''
            Creates scatter plot items for each plotDataItem using the
            point closest to the center of the radius rect
        '''

        # Get the center and ranges of the radius rect
        xmin, xmax = radius_rect.left(), radius_rect.right()
        ymin, ymax = radius_rect.top(), radius_rect.bottom()
        xcenter, ycenter = radius_rect.center().x(), radius_rect.center().y()

        # Find the closest point to the center of the radius rect for
        # each point and create a scatter plot item for it
        hover_pts = []
        for pdi in pdis:
            # Get data and line info
            x, y = pdi.getData()
            pen = pdi.opts['pen']

            # Clip data to relevant x region (assumes x is monotonically increasing)
            start = bisect_left(x, xmin)
            end = bisect_right(x, xmax)
            x = x[start:end+1]
            y = y[start:end+1]

            # Skip if points list is empty
            if len(x) <= 0:
                continue

            # Clip points out of max y range
            mask = np.logical_and(y>=ymin, y<=ymax)
            x = x[mask]
            y = y[mask]
            
            # Skip if points list is empty here
            if len(x) <= 0:
                continue

            # Calculate the euclidean distance between the cursor position
            # and all the points that have been left after clipping
            dx, dy = self.getViewBox().viewPixelSize() # Need to scale to pixels
            dist = ((x-xcenter)/dx)**2 + ((y-ycenter)/dy)**2
            dist = np.sqrt(dist)

            # Find the point with the smallest euclidean distance
            if len(dist) <= 1:
                min_index = 0
            else:
                min_index = np.argmin(dist)

            # Create array of single point
            x_masked = [x[min_index]]
            y_masked = [y[min_index]]

            # Create scatter plot item with a color that's slightly darker
            color = pen.color().darker(115)
            outline = color.darker(110)
            pen = pg.mkPen(outline)
            brush = pg.mkBrush(color)

            scatter = LabeledScatter(self.format_funcs, x_masked, y_masked, size=10, pen=pen, brush=brush)

            scatter.setParentItem(self)
            hover_pts.append(scatter)
        
        return hover_pts
