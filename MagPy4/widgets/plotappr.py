from numpy.core.fromnumeric import trace
from MagPy4.extraticklabels import AddTickLabels
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from ..plotbase import DateAxis
import re
from datetime import datetime, timedelta
import numpy as np
from ..layouttools import BaseLayout

import pyqtgraph as pg
import functools

class PlotAppearanceUI(BaseLayout):
    def setupUI(self, Frame, plt_grd):
        Frame.setWindowTitle('Plot Appearance')
        Frame.resize(300, 200)

        # Set up tab widget in layout
        layout = QtWidgets.QGridLayout(Frame)
        self.layout = layout
        tw = QtWidgets.QTabWidget()
        self.tabWidget = tw
        layout.addWidget(tw, 3, 0, 1, 4)

        # Set up UI for setting plot trace colors, line style, thickness, etc.
        plotsInfo = plt_grd.get_traces()
        tracePropFrame, trace_count = self.getTracePropFrame(plotsInfo)

        # If there are a lot of traces, wrap the trace properties frame
        # inside a scroll area
        tracePropScroll = self.wrapTracePropFrame(tracePropFrame)
        if trace_count > 0:
            tw.addTab(tracePropScroll, 'Trace Properties')

        # # Set up tick intervals widget
        tickIntWidget = MagPyTickIntervals(plt_grd)
        tw.addTab(tickIntWidget, 'Tick Spacing')

        # Set up label properties widget
        self.lblPropWidget = LabelAppear(plt_grd)
        tw.addTab(self.lblPropWidget, 'Label Sizes')

        # Set up tick labels widget
        self.tickLabels = AddTickLabels(plt_grd, plt_grd.window)
        if self.tickLabels.valid:
            tw.addTab(self.tickLabels, 'Tick Labels')

    def setTab(self, tab_num):
        self.tabWidget.setCurrentIndex(tab_num)

    def getTracePropFrame(self, plotInfos):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        self.styleGrps, self.widthGrps, self.colorGrps = [], [], []
        count = 0
        for index, info in plotInfos:
            plotFrame, elems = self.getPlotPropFrame(index, len(info))

            # Hide if special plot
            if len(info) == 0:
                plotFrame.setVisible(False)
            else:
                count += len(info)

            styleBoxes, widthBoxes, colorBtns = elems
            self.styleGrps.append(styleBoxes)
            self.widthGrps.append(widthBoxes)
            self.colorGrps.append(colorBtns)
            layout.addWidget(plotFrame)
        layout.addStretch()

        return frame, count

    def getPlotPropFrame(self, index, numTraces):
        # Build the properties frame for a single plot item
        frame = QtWidgets.QGroupBox(f'Plot {index+1}:')
        layout = QtWidgets.QGridLayout(frame)
        layout.setHorizontalSpacing(12)

        # Build header layout
        labels = [f'{label}:' for label in ['#', 'Style', 'Width', 'Color']]
        for i, label in enumerate(labels):
            layout.addWidget(QtWidgets.QLabel(label), 0, i, 1, 1)

        # Create line properties settings layout for each line
        styleBoxes, widthBoxes, colorBtns = [], [], []
        for index in range(0, numTraces):
            elems = self.getLinePropLt(index)
            for i, elem in enumerate(elems):
                layout.addWidget(elem, index+1, i, 1, 1)

            # Store line property elements
            lbl, style, width, color = elems
            styleBoxes.append(style)
            widthBoxes.append(width)
            colorBtns.append(color)

        return frame, (styleBoxes, widthBoxes, colorBtns)

    def getLinePropLt(self, index):
        # Line index label
        label = QtWidgets.QLabel(f'{index+1}:')
        label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Create all elements for choosing line style
        lineStyle = QtWidgets.QComboBox()
        for t in ['Solid', 'Dashed', 'Dotted', 'DashDot']:
            lineStyle.addItem(t)

        # Create all elements for choosing line thickness
        lineWidth = QtWidgets.QDoubleSpinBox()
        lineWidth.setMinimum(1)
        lineWidth.setMaximum(5)
        lineWidth.setSingleStep(0.5)

        # Create elements for choosing line color
        colorBtn = QtWidgets.QPushButton()
        colorBtn.setCursor(QtCore.Qt.ClosedHandCursor)

        return (label, lineStyle, lineWidth, colorBtn)

    def wrapTracePropFrame(self, widgetFrame):
        # Create a scroll area and set its dimensions
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(widgetFrame)
        scrollArea.setWidgetResizable(True)
        scrollArea.setMinimumHeight(min(max(widgetFrame.sizeHint().height(), 300), 500))
        scrollArea.setMinimumWidth(350)
        scrollArea.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Set style properties to match tab widget's background
        scrollArea.setStyleSheet('QScrollArea {background-color: transparent;}')
        widgetFrame.setStyleSheet('QFrame {background-color: white;}')

        # Hide bottom scroll bar
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        return scrollArea

class PlotAppearance(QtWidgets.QFrame, PlotAppearanceUI):
    colorsChanged = QtCore.pyqtSignal(object)
    def __init__(self, pltGrd, parent=None):
        super(PlotAppearance, self).__init__(parent)
        self.ui = PlotAppearanceUI()
        self.pltGrd = pltGrd
        self.plotItems = pltGrd.get_plots()

        # Get plots' trace/label infos and use to setup/initialize UI elements
        plotsInfo = self.getPlotLineInfos()
        self.ui.setupUI(self, pltGrd)
        self.initVars(plotsInfo)

        # Connect line width modifiers to function
        grps = (plotsInfo, self.ui.styleGrps, self.ui.widthGrps, self.ui.colorGrps)
        for (index, info), styleGrp, widthGrp, colorGrp in zip(*grps):
            traceIndex = 0
            for line, style, width, color in zip(info, styleGrp, widthGrp, colorGrp):
                line['plotIndex'] = index
                line['traceIndex'] = traceIndex
                widthFunc = functools.partial(self.updateLineWidth, line)
                width.valueChanged.connect(widthFunc)

                styleFunc = functools.partial(self.updateLineStyle, line)
                style.currentTextChanged.connect(styleFunc)

                colorFunc = functools.partial(self.openColorSelect, color, line)
                color.clicked.connect(colorFunc)
                traceIndex += 1

        self.plotInfo = plotsInfo

    def getPlotLineInfos(self):
        # Creates list of per-plot lists containing tuples of pens and a list of
        # data items within the given plot that correspond to it
        # ex: [ [(pen1, [plotDataItem1...]), (pen2, [..])] , ... , ]
        n = len(self.plotItems)
        plotInfos = [(i, plt.getLineInfo()) for i, plt in zip(range(n), self.plotItems)]
        return plotInfos

    def getPenStyle(self, pen):
        # Extract style information from pen
        pen = pg.mkPen(pen)
        style = pen.style()
        width = pen.widthF()
        color = pen.color()

        return (style, width, color)

    def initVars(self, plotsInfo):
        # Initializes all values in UI elements with current plot properties
        if self.plotItems == None:
            return

        # Maps Qt Style flag to string
        styleMap = {
            QtCore.Qt.DashLine: 'Dashed',
            QtCore.Qt.DotLine: 'Dotted',
            QtCore.Qt.DashDotLine:'DashDot',
        }

        # Iterate over plots and their traces to initialize pen settings
        for index, info in plotsInfo:
            lineNum = 0
            for lineInfo in info:
                pen = lineInfo['pen']
                style, width, color = self.getPenStyle(pen)

                # Map the Qt Pen style to a string
                if style in styleMap:
                    styleStr = styleMap[style]
                else:
                    styleStr = 'Solid'

                # Set UI elements with values
                self.ui.styleGrps[index][lineNum].setCurrentText(styleStr)
                self.ui.widthGrps[index][lineNum].setValue(width)
                self.setButtonColor(self.ui.colorGrps[index][lineNum], color)

                lineNum += 1

    def updateLineWidth(self, lineInfo, val):
        # Extract line and its pen item
        line = lineInfo['item']
        pen = pg.mkPen(lineInfo['pen'])

        # Set new pen width
        pen.setWidthF(val)
        line.setPen(pen)
        lineInfo['pen'] = pen

        # Apply any other persistent changes TODO:
        self.setChangesPersistent(lineInfo)

    def updateLineStyle(self, lineInfo, styleStr):
        # Get style object from name
        if styleStr == 'Dashed':
            style = QtCore.Qt.DashLine
        elif styleStr == 'Dotted':
            style = QtCore.Qt.DotLine
        elif styleStr == 'DashDot':
            style = QtCore.Qt.DashDotLine
        else:
            style = QtCore.Qt.SolidLine

        # Update pens for selected plots
        line = lineInfo['item']
        pen = pg.mkPen(lineInfo['pen'])
        pen.setStyle(style)
        line.setPen(pen)
        lineInfo['pen'] = pen

        # TODO: Apply other persistent changes
        self.setChangesPersistent(lineInfo)

    def openColorSelect(self, btn, lineInfo):
        # Open color selection dialog and connect to line color update function
        clrDialog = QtGui.QColorDialog(self)
        clrDialog.show()
        pen = pg.mkPen(lineInfo['pen']).color()
        clrDialog.setCurrentColor(pen)
        clrDialog.colorSelected.connect(functools.partial(self.setLineColor, btn, lineInfo))

    def setLineColor(self, btn, lineInfo, color):
        # Update pen color of every trace item
        pen = pg.mkPen(lineInfo['pen'])
        pen.setColor(color)
        lineInfo['pen'] = pen
        old_color = pen.color()

        line = lineInfo['item']
        line.setPen(pen)

        # Match the color selection button to the selected color
        self.setButtonColor(btn, color)

        # Set the title colors to match, if implemented
        self.setChangesPersistent(lineInfo)

        plot_index = lineInfo['plotIndex']
        plot = self.plotItems[plot_index]
        name = lineInfo['name']
        self.pltGrd.update_label_color(plot, name, color.name())

        signal = (plot, name, (old_color, color))
        self.colorsChanged.emit(signal)

    def setButtonColor(self, cs, color):
        styleSheet = f'''
            * {{
                background: {color.name()};
            }}
        '''
        cs.setStyleSheet(styleSheet)
        cs.show()

    # Placeholder func to be implemented to make line changes persistent
    # between changes to the plot traces
    def setChangesPersistent(self, line):
        pass

    def getPenList(self):
        # Returns a list of the pen lists for each plot in plotItems
        pltInfos = self.getPlotsInfo()
        penList = []
        # For each plot, extract its pen list from its plotInfo
        for pltInfo in pltInfos:
            pltPens = []
            for pen, ll in pltInfo:
                pltPens.append(pen)
            penList.append(pltPens)
        return penList

    def allowLeftAxisEditing(self):
        # Allow left axis editing if not applying change to a color plot
        return True

class SectionLineEdit(QtWidgets.QLineEdit):
    initialSelect = QtCore.pyqtSignal()
    def __init__(self, select_func=None, *args, **kwargs):
        self.select_func = select_func
        super().__init__(*args, **kwargs)
        self.cursorPositionChanged.connect(self.select_section)
        self.editingFinished.connect(self.end_select)

    def start_editing(self, pos):
        if not self.hasSelectedText() and pos != len(self.text()):
            self.initialSelect.emit()
    
    def select_section(self, old_pos, new_pos):
        if self.hasSelectedText():
            return

        rng = self.select_func(old_pos, new_pos)
        if rng:
            start, end = rng
            self.blockSignals(True)
            self.setSelection(start, end)
            self.blockSignals(False)
    
    def end_select(self):
        self.deselect()

class TimeIntBox(QtWidgets.QAbstractSpinBox):
    ''' Input box for specifying time intervals '''
    def __init__(self):
        super().__init__()
        # Default keys and values
        self.keys = {
            'days' : 0,
            'hours' : 0,
            'min' : 0,
            'sec' : 0,
        }

        # Regular expressions to match keyword pairs
        self.exprs = {}
        for key in self.keys:
            expr = f'{key}=[0-9]+'
            self.exprs[key] = expr

        # Key ordering
        self.order = ['days', 'hours', 'min', 'sec']
        self.factors = [24*60*60, 60*60, 60, 1]
        self.editingFinished.connect(self.validateAfterReturn)

        # Set default text and minimum widths
        self.clear()
        self.setMinimumWidth(275)

        # Set line editor and connect clicks to section selector
        self.setLineEdit(SectionLineEdit(self.get_section))
        self.validateAfterReturn()
    
    def get_section(self, old_pos, pos):
        text = self.lineEdit().text()
        expr = '[A-z]+=[0-9]+'
        grps = list(re.finditer(expr, text))

        i = max(np.sign(pos - old_pos), 1)

        rng = None
        for grp in grps[::i]:
            start = grp.start()
            end = grp.end()
            if pos >= start and pos < end:
                subtext = text[start:end]
                select_start = start
                lft, rght = subtext.split('=')
                select_start += len(lft) + 1
                select_len = len(rght)
                rng = (select_start, select_len)
                break

        return rng

    def interpretText(self):
        ''' Maps text to a timedelta value '''
        # Get key/value pairs from text
        text = self.lineEdit().text()
        pairs = {k:v for k, v in self.keys.items()}
        validExpr, keys = self.extractElements(text)
        pairs.update(keys)

        # Map all key/value pairs to seconds quantities
        seconds = 0
        for key, factor in zip(self.order, self.factors):
            base_val = pairs[key]
            base_val *= factor
            seconds += base_val

        # Create a timedelta object from the sum of all seconds quantities
        td = timedelta(seconds=seconds)
        return td

    def getValue(self):
        ''' Return timedelta represented by text '''
        return self.interpretText()

    def extractElements(self, text):
        ''' Extracts key/value pairs from text '''
        # Split entries by commas
        if ',' not in text:
            return True, {}
        else:
            split_text = text.split(',')

        # Try to match with one of the keyword regular expressions
        invalidExpr = False
        matching_keys = {}
        for entry in split_text:
            entry = entry.strip(' ')

            for key in self.exprs:
                expr = self.exprs[key]
                # If expression matches, save key/value pair in dictionary
                if re.fullmatch(expr, entry):
                    value = entry.split('=')[1]
                    value = int(value)
                    matching_keys[key] = value
                else:
                    invalidExpr = True

        # Return whether any invalid expressions were found and
        # a dictionary of matching keys and their values
        return (invalidExpr, matching_keys)

    def clear(self):
        # Sets text to display key=0 for all keys
        text = self.formatPairs(self.keys)
        self.lineEdit().setText(text)

    def fixup(self, text):
        ''' Adjusts text if not in valid form '''
        # Get matching keys
        invalid_expr, match_keys = self.extractElements(text)
        pairs = {key:value for key, value in self.keys.items()}
        if len(match_keys) == 0: # No keys match at all, use defaults
            pairs = pairs
        else: # Some keys match, display only theses
            pairs.update(match_keys)

        # Format key value pairs into a string using only
        # the valid key/value pairs from the user and other
        # values reset to zero
        new_text = self.formatPairs(pairs)
        self.lineEdit().setText(new_text)

    def formatPairs(self, pairs):
        ''' Formats items in pairs dictionary into
            a string in 'key1=val1, key2=val2, ...' format
        '''
        text = []
        for key in pairs:
            value = pairs[key]
            substr = f'{key}={value}'
            text.append(substr)

        text = ', '.join(text)
        return text

    def stepBy(self, steps):
        ''' Increment/decrement timedelta object by {steps} seconds '''
        td = self.interpretText()
        td = td + timedelta(seconds=steps)
        self.setValue(td)

    def setValue(self, td):
        ''' Sets key/value pair values based on timedelta's total seconds '''
        td = max(timedelta(seconds=0), td)
        remaining_td = td
        pairs = {}
        # For each key and corresponding multiplying factor
        for key, factor in zip(self.order, self.factors):
            # Get the quantity represented by the timedelta
            # and update the current timedelta to the remaining
            # seconds
            amt, remaining_td = divmod(remaining_td, timedelta(seconds=factor))
            amt = int(amt)
            pairs[key] = amt

        # Update text with new values
        text = self.formatPairs(pairs)
        self.lineEdit().setText(text)

    def stepEnabled(self):
        ''' Step up and step down should both be enabled '''
        stepUp = QtWidgets.QAbstractSpinBox.StepDownEnabled
        stepDown = QtWidgets.QAbstractSpinBox.StepUpEnabled
        return  stepUp | stepDown

    def minimum(self):
        ''' Minimum timedelta is zero seconds '''
        minimum = timedelta(seconds=0)
        return minimum

    def maximum(self):
        ''' Maximum timedelta is system max '''
        maximum = timedelta.max
        return maximum

    def validateAfterReturn(self):
        ''' Fixup text if validator does not return 'Acceptable' '''
        lineEdit = self.lineEdit()
        text = lineEdit.text()
        pos = lineEdit.cursorPosition()
        val, text, pos = self.validate(text, pos)
        if val != QtGui.QValidator.Acceptable:
            self.fixup(text)

    def validate(self, text, pos):
        ''' Validate text while cursor is at given position '''
        invalid_expr, match_keys = self.extractElements(text)
        if len(match_keys) == 0: # No matching keys
            val = QtGui.QValidator.Invalid
        elif invalid_expr: # Some matching keys, others invalid
            val = QtGui.QValidator.Intermediate
        else: # All matching keys
            val = QtGui.QValidator.Acceptable
        return (val, text, pos)

class TickIntBox(QtWidgets.QDoubleSpinBox):
    def __init__(self):
        super().__init__()

    def setLogScale(self, val):
        ''' Adjust prefix and decimal places based
            on whether displaying log scale or not
        '''
        if val:
            self.setPrefix('10^')
            self.setDecimals(0)
        else:
            self.setPrefix('')
            self.setDecimals(2)

    def getValue(self):
        return self.value()

class TickWidget(QtWidgets.QGroupBox):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, plots, axis, *args):
        # Map axis position to counterpart edge value
        self.edge_map = {'left':'right', 'bottom':'top'}
        self.name = axis # Axis edge
        self.plots = plots # Plots affected by this widget
        super().__init__(*args)
        self.layout = self.setupUI()

        # Initialize tick interval values with any values
        # currently set for the plot items
        self.initValues()

        # Link buttons to actions
        self.applyBtn.clicked.connect(self.apply)
        self.defaultBtn.clicked.connect(self.default)

    def setupUI(self):
        layout = QtWidgets.QGridLayout(self)

        # Set up major and minor tick boxes
        self.majorBox = self.getWidget()
        self.minorBox = self.getWidget()

        boxLt = QtWidgets.QGridLayout()
        row = 0
        for label, box in zip(['Major:', 'Minor:'], [self.majorBox, self.minorBox]):
            labelItem = QtWidgets.QLabel(label)
            labelItem.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            boxLt.addWidget(labelItem, row, 0, 1, 1)
            boxLt.addWidget(box, row, 1, 1, 1)
            row += 1

        layout.addLayout(boxLt, 0, 0, 1, 1)

        # Set up default and apply buttons
        self.defaultBtn = QtWidgets.QPushButton('Default')
        self.applyBtn = QtWidgets.QPushButton('Apply')

        btnLt = QtWidgets.QVBoxLayout()
        for btn in [self.defaultBtn, self.applyBtn]:
            btnLt.addWidget(btn)

        layout.addLayout(btnLt, 0, 1, 1, 1)

        return layout

    def getWidget(self):
        ''' Return the widget to use for the tick interval '''
        intBox = TickIntBox()
        return intBox

    def getValue(self):
        ''' Return the tick spacing value(s) '''
        # Extract values from UI
        major = self.majorBox.getValue()
        minor = self.minorBox.getValue()

        # Add to list if not set to minimum
        diff = []
        if major != self.majorBox.minimum():
            diff.append(major)
        if minor != self.minorBox.minimum():
            diff.append(minor)

        # Ignore if both are not valid intervals
        if len(diff) == 0:
            diff = None

        return diff

    def setValue(self, val):
        ''' Interprets interval value and sets major/min boxes
            accordingly
        '''
        major, minor = None, None

        # Map from list or only set major if it is a single number
        if isinstance(val, (np.ndarray, list)):
            if len(val) > 0:
                major = val[0]
            if len(val) > 1:
                minor = val[1]
        else:
            major = val

        # Set box values if available
        if major:
            self.majorBox.setValue(major)

        if minor:
            self.minorBox.setValue(minor)

    def formatter(self):
        ''' No tick label formatter is used by default '''
        return None

    def initValues(self):
        ''' Initialize tick values and logScale settings
            from plot settings '''
        # Set value if plot has tickDiff set
        ax = self.plots[-1].getAxis(self.name)
        if ax.tickDiff is not None:
            self.setValue(ax.tickDiff)

        # Set log scaling if plots are in log scale
        if ax.logMode:
            self.majorBox.setLogScale(True)
            self.minorBox.setLogScale(True)

    def apply(self):
        ''' Set tick spacing for plots '''
        # Extract value from UI
        value = self.getValue()

        # Ignore if both are not valid
        if len(value) == 0:
            return

        # Set tick spacing for both axes for all plots
        for plot in self.plots:
            edges = [self.name, self.edge_map[self.name]]
            for edge in edges:
                ax = plot.getAxis(edge)
                ax.setCstmTickSpacing(value)

        self.valueChanged.emit(value)

    def default(self):
        ''' Reset tick spacing for plots '''
        # Get both axes for each plot and reset tick spacing
        for plot in self.plots:
            edges = [self.name, self.edge_map[self.name]]
            for edge in edges:
                ax = plot.getAxis(edge)
                ax.resetTickSpacing()

        self.valueChanged.emit(None)

class TimeTickWidget(TickWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        format_lt, self.formatBox = self.buildTimeLblLt()
        self.layout.addLayout(format_lt, 1, 0, 1, 1)
        self.formatBox.currentTextChanged.connect(self.setLabels)
        self.initFormat()

    def formatter(self):
        ''' Returns the label formatter object '''
        return self.formatBox

    def getWidget(self):
        ''' Returns a time interval widget '''
        return TimeIntBox()

    def buildTimeLblLt(self):
        # Get time label formats
        fmts = DateAxis(orientation='bottom', epoch='J2000').get_label_modes()
        fmts = ['Default'] + fmts

        # Build box and layout elements
        layout = QtWidgets.QHBoxLayout()
        fmtBox = QtWidgets.QComboBox()
        fmtBox.addItems(fmts)

        label = QtWidgets.QLabel('Label Format: ')
        label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        layout.addWidget(label)
        layout.addWidget(fmtBox)
        return layout, fmtBox

    def setLabels(self, val):
        ''' Update tick label format for DateTime axes '''
        if val == 'Default':
            for plot in self.plots:
                ax = plot.getAxis(self.name)
                ax.set_label_fmt(None)
        else:
            for plot in self.plots:
                ax = plot.getAxis(self.name)
                ax.set_label_fmt(val)

    def initFormat(self):
        ''' Set label format if default is not used for
            current axis
        '''
        ax = self.plots[-1].getAxis(self.name)
        if ax.get_label_fmt():
            fmt = ax.get_label_fmt()
            self.formatBox.setCurrentText(fmt)

class TickIntervals(QtWidgets.QFrame):
    def __init__(self, pltGrd, parent=None, links=None):
        self.pltGrd = pltGrd
        super(TickIntervals, self).__init__(parent)

        # Update links to include all plots if links is empty or None
        plots = self.pltGrd.get_plots()
        links = self.pltGrd.get_links()
        if links is None or len(links) == 0:
            links = [[i for i in range(len(plots))]]
        self.links = links

        self.setupUI(self.pltGrd.get_plots(), self.links)

    def setupUI(self, plotItems, links):
        self.setWindowTitle('Set Tick Spacing')
        self.resize(200, 100)
        layout = QtWidgets.QVBoxLayout(self)

        # Assemble left axis tick widgets for each link group
        for linkGrp in links:
            plotGrp = [plotItems[i] for i in linkGrp]
            if len(links) != 1:
                label = f'Linked Axes {str(linkGrp)}'
            else:
                label = f'Left Axis'
            widget = TickWidget(plotGrp, 'left', label)
            layout.addWidget(widget)

        # Determine what type of tick widget to use for bottom axes
        axType = plotItems[-1].getAxis('bottom').axisType()
        label = 'Bottom Axis'
        if axType == 'DateTime':
            widget = TimeTickWidget(plotItems, 'bottom', label)
        else:
            widget = TickWidget(plotItems, 'bottom', label)
        layout.addWidget(widget)
        layout.addStretch()

    def additionalUpdates(self, value):
        pass

class MagPyTickIntervals(TickIntervals):
    def __init__(self, pltGrd):
        self.pltGrd = pltGrd
        super().__init__(pltGrd)

class LabelAppearUI(BaseLayout):
    def setupUI(self, Frame):
        frameLt = QtWidgets.QVBoxLayout(Frame)
        layout = QtWidgets.QGridLayout()
        self.layout = layout

        # Font size label setup
        self.titleSzLbl = QtWidgets.QLabel('Title Size: ')
        self.axisLblSzLbl = QtWidgets.QLabel('Axis Label Size: ')
        self.tickLblSzLbl = QtWidgets.QLabel('Tick Label Size: ')

        # Title, axis label, and tick label spinboxes setup
        self.titleSzBox = QtWidgets.QSpinBox()
        self.titleSzBox.setMinimum(5)
        self.titleSzBox.setMaximum(30)

        self.axisLblSzBox = QtWidgets.QSpinBox()
        self.axisLblSzBox.setMinimum(5)
        self.axisLblSzBox.setMaximum(25)

        self.tickLblSzBox = QtWidgets.QSpinBox()
        self.tickLblSzBox.setMinimum(5)
        self.tickLblSzBox.setMaximum(25)

        lbls = [self.titleSzLbl, self.axisLblSzLbl, self.tickLblSzLbl]
        boxes = [self.titleSzBox, self.axisLblSzBox, self.tickLblSzBox]
        row = 0
        for lbl, box in zip(lbls, boxes):
            layout.addWidget(lbl, row, 0, 1, 1, QtCore.Qt.AlignLeft)
            layout.addWidget(box, row, 1, 1, 1, QtCore.Qt.AlignLeft)
            lbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            box.setFixedWidth(100)
            # Fill in empty space to right
            spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.MinimumExpanding)
            layout.addItem(spacer, row, 2, 1, 1)
            row += 1

        # Add in default box for axis label sizes in main window
        spacer = layout.itemAtPosition(1, 2)
        layout.removeItem(spacer)
        self.defaultBtn = QtWidgets.QPushButton('Default')
        self.defaultBtn.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        layout.addWidget(self.defaultBtn, 1, 2, 1, 1)

        frameLt.addLayout(layout)
        frameLt.addStretch()

class LabelAppear(QtWidgets.QFrame, LabelAppearUI):
    def __init__(self, pltGrd, parent=None):
        super(LabelAppear, self).__init__(parent)
        self.ui = LabelAppearUI()
        self.pltGrd = pltGrd

        # Get plots' trace/label infos and use to setup/initialize UI elements
        self.ui.setupUI(self)
        self.initVars()

        # Connect spinbox changes to functions
        self.ui.titleSzBox.valueChanged.connect(self.changeTitleSize)
        self.ui.axisLblSzBox.valueChanged.connect(self.changeAxisLblSize)
        self.ui.tickLblSzBox.valueChanged.connect(self.changeTickLblSize)
        self.ui.defaultBtn.clicked.connect(self.resizeMainAxisLbls)

    def resizeMainAxisLbls(self):
        # Unlock label resize settings and resize plot grid so they
        # are reverted to default sizes
        pass
 
    def initVars(self):
        plots = self.pltGrd.get_plots()
        plt = plots[-1]
        # Get title font size to initialize spin box, disable if no title
        titleSize = plt.titleLabel.opts['size'][:-2] # Strip pts part of string
        if plt.titleLabel.text == '':
            for elem in [self.ui.titleSzBox, self.ui.titleSzLbl]:
                self.ui.layout.removeWidget(elem)
                elem.deleteLater()
        else:
            self.ui.titleSzBox.setValue(int(titleSize))

        # Initialize axis label font size, disable if no axis label
        self.ui.axisLblSzBox.blockSignals(True) # Prevent plot updates temporarily
        if plt.getAxis('bottom').label.toPlainText() == '':
            self.ui.axisLblSzBox.setEnabled(False)
            self.ui.axisLblSzLbl.setEnabled(False)
        elif 'font-size' in plt.getAxis('bottom').labelStyle:
            axisLblSize = plt.getAxis('bottom').labelStyle['font-size'][:-2]
            self.ui.axisLblSzBox.setValue(int(axisLblSize))
        else:
            self.ui.axisLblSzBox.setValue(11) # Default axis label font size
        self.ui.axisLblSzBox.blockSignals(False)

        # Initialize tick label font size
        self.ui.tickLblSzBox.blockSignals(True)
        axis = plt.getAxis('bottom')
        if axis.getTickFont() == None: # No custom font, use default
            self.ui.tickLblSzBox.setValue(11)
        else:
            tickFont = axis.getTickFont()
            tickSize = tickFont.pointSize()
            if tickSize < 0: # No point size set, use default
                tickSize = 11
            self.ui.tickLblSzBox.setValue(int(tickSize))
        self.ui.tickLblSzBox.blockSignals(False)

    def changeTitleSize(self, val):
        plots = self.pltGrd.get_plots()
        for plt in plots:
            plt.titleLabel.setText(plt.titleLabel.text, size=str(val)+'pt')

    def changeAxisLblSize(self, val):
        # Set stacked label sizes for main window plot grid and lock label size
        # Update every plot's label sizes for every axis
        plots = self.pltGrd.get_plots()
        for plt in plots:
            for ax in [plt.getAxis('bottom'), plt.getAxis('left')]:
                if ax.label.toPlainText() == '': # Do not change size if no label
                    continue
                # Convert new value to apprp string and update label info
                sizeStr = str(val) + 'pt'
                ax.labelStyle = {'font-size':sizeStr}
                ax.label.setHtml(ax.labelString()) # Uses updated HTML string

    def changeTickLblSize(self, val):
        # Update every axes' tick label sizes
        self.pltGrd.set_tick_text_size(val)
    
    def adjustTickHeights(self, axis, tickFont):
        # Adjust vertical spacing reserved for bottom ticks if necessary
        mets = QtGui.QFontMetrics(tickFont)
        ht = mets.boundingRect('AJOW').height() # Tall letters
        if ht > 18 and axis.orientation == 'bottom':
            ht = max(int(ht*.25), 5)
            axis.setStyle(tickTextOffset=ht)
        elif axis.orientation == 'bottom':
            axis.setStyle(tickTextOffset=2)
