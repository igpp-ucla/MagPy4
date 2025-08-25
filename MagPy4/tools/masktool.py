from ..qtinterface.layouttools import BaseLayout
from . import waveanalysis
from ..plotbase import MagPyPlotItem, StackedAxisLabel
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
from scipy import ndimage
import numpy as np

class ToolSpecificLt(BaseLayout):
    def __init__(self):
        self.logMode = False
        super().__init__()
    
    def getMaskLt(self, valRng, logMode=False, frame=None):
        if frame:
            layout = QtWidgets.QGridLayout(frame)
        else:
            layout = QtWidgets.QGridLayout()

        # Set up boxes for specifying mask ranges
        self.minMaskCheck = QtWidgets.QCheckBox(' Greater than: ')
        self.maxMaskCheck = QtWidgets.QCheckBox(' Less than: ')

        self.minMaskBox = QtWidgets.QDoubleSpinBox()
        self.maxMaskBox = QtWidgets.QDoubleSpinBox()

        # Op box if both are selected
        self.subMaskOp = QtWidgets.QComboBox()
        self.subMaskOp.addItems(['AND', 'OR'])

        layout.addWidget(self.minMaskCheck, 0, 0, 1, 1)
        layout.addWidget(self.minMaskBox, 0, 1, 1, 1)
        layout.addWidget(self.maxMaskCheck, 1, 0, 1, 1)
        layout.addWidget(self.maxMaskBox, 1, 1, 1, 1)
        self.opLbl = self.addPair(layout, 'Mask Operation: ', self.subMaskOp, 2, 0, 1, 1)
        for elem in [self.opLbl, self.subMaskOp]:
            elem.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Set box settings
        minVal, maxVal = valRng
        if logMode:
            for box in [self.minMaskBox, self.maxMaskBox]:
                box.setPrefix('10^')
            self.logMode = True

        for box in [self.minMaskBox, self.maxMaskBox]:
            box.setMinimum(minVal)
            box.setMaximum(maxVal)
            box.setDecimals(3)

        # Hide op box when both are not set        
        for chk in [self.minMaskCheck, self.maxMaskCheck]:
            chk.toggled.connect(self.hideOpBox)
            chk.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.hideOpBox()

        return layout

    def getMaskRanges(self):
        # If min/max boxes are checked, return the box value or return None
        # otherwise
        if not self.minMaskCheck.isChecked():
            minVal = None
        else:
            minVal = self.minMaskBox.value()

        if not self.maxMaskCheck.isChecked():
            maxVal = None
        else:
            maxVal = self.maxMaskBox.value()

        # Scale if ranges are logarithmic
        if self.logMode:
            minVal = 10 ** minVal if minVal is not None else minVal
            maxVal = 10 ** maxVal if maxVal is not None else maxVal

        # Also return the binary operation to apply between the two
        # masks if both are set
        opVal = self.subMaskOp.currentText()

        return minVal, maxVal, opVal

    def hideOpBox(self):
        # Show binary op combobox only if both min and max boxes are checked
        minChecked = self.minMaskCheck.isChecked()
        maxChecked = self.maxMaskCheck.isChecked()
        if minChecked and maxChecked:
            self.subMaskOp.setVisible(True)
            self.opLbl.setVisible(True)
        else:
            self.subMaskOp.setVisible(False)
            self.opLbl.setVisible(False)

    def getVarInfo(self):
        return None

class MaskToolUI(BaseLayout):
    def setupUI(self, maskFrame, plotTool, plotType, waveNames=None):
        self.maskFrame = maskFrame
        maskFrame.setWindowTitle('Mask Tool')
        layout = QtWidgets.QGridLayout(maskFrame)
        self.layout = layout

        # Get default min/max values to use in the mask value range setter
        minVal, maxVal = 0, 1
        logMode = False
        if plotType == 'Spectra':
            minVal = plotTool.ui.valueMin.minimum()
            maxVal = plotTool.ui.valueMax.maximum()
            logMode = True
        elif plotType in ['Coherence', 'Phase']:
            if plotType == 'Coherence':
                minVal, maxVal = 0, 1.0
            else:
                minVal, maxVal = -180, 180
        else:
            valRange = plotTool.defParams[plotType][0]
            if valRange is not None:
                minVal, maxVal = valRange
            else:
                minVal, maxVal = -18, 18
            if plotType in plotTool.plotGroups['Power']:
                logMode = True

        # Set up mask values calculator
        groupFrame = QtWidgets.QGroupBox('Mask Values')
        self.toolMaskLt = ToolSpecificLt()
        toolMaskSubLt = self.toolMaskLt.getMaskLt((minVal, maxVal), logMode, groupFrame)

        # Mask properties layout
        maskPropLt = self.setupMaskSettingsLt()

        # Update button
        self.updateBtn = QtWidgets.QPushButton(' Plot ')

        settingsLt = QtWidgets.QHBoxLayout()
        for w in [groupFrame, maskPropLt, self.updateBtn]:
            settingsLt.addWidget(w)
        settingsLt.setAlignment(self.updateBtn, QtCore.Qt.AlignBottom)

        # Apply coherence mask to other plots section
        self.applyRow = QtWidgets.QHBoxLayout()
        self.applyRowLbl = QtWidgets.QLabel('Apply coherence mask to:')
        self.applyTarget = QtWidgets.QComboBox()
        self.applyTarget.addItems(['—', 'Dynamic Spectra', 'Dynamic Wave Analysis'])

        self.waveFunc = QtWidgets.QComboBox()
        self.waveFunc.setVisible(False)
        if waveNames:
            self.waveFunc.addItems(list(waveNames))

        def _toggle_wave(txt):
            self.waveFunc.setVisible(txt == 'Dynamic Wave Analysis')
        self.applyTarget.currentTextChanged.connect(_toggle_wave)

        self.applyBtn = QtWidgets.QPushButton('Plot masked')
        self.applyBtn.setEnabled(False)

        self.applyRow.addWidget(self.applyRowLbl)
        self.applyRow.addWidget(self.applyTarget, 1)
        self.applyRow.addWidget(self.waveFunc, 2)
        self.applyRow.addWidget(self.applyBtn)

        # Coherence mask only
        layout.addLayout(settingsLt, 0, 0, 1, 2, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        layout.addLayout(self.applyRow, 1, 0, 1, 2, QtCore.Qt.AlignLeft)

        # Plot graphics grid
        self.glw = self.getGraphicsGrid()
        self.gview.setVisible(False)
        layout.addWidget(self.gview, 2, 0, 1, 2)
        
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 1)
        
    def setupMaskSettingsLt(self):
        frame = QtWidgets.QGroupBox('Mask Properties')
        frame.resize(100, 50)
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(10)

        wrapLt = QtWidgets.QVBoxLayout(frame)
        wrapLt.addLayout(layout)
        wrapLt.setAlignment(QtCore.Qt.AlignTop)

        # Set up button for setting mask color
        self.colorBox = QtWidgets.QPushButton()
        self.colorBox.clicked.connect(self.openColorSelect)

        # Default mask color is black
        self.maskColor = (0, 0, 0)
        self.setMaskColor(QtGui.QColor(0, 0, 0))

        colorLt = QtWidgets.QGridLayout()
        colorLt.setContentsMargins(0, 0, 0, 0)
        self.addPair(colorLt, 'Color:  ', self.colorBox, 0, 0, 1, 1)

        # Mask outline box
        self.outlineCheck = QtWidgets.QCheckBox(' Outline Only')

        # Filtering check box
        filterLt = QtWidgets.QVBoxLayout()
        self.filterCheck = QtWidgets.QCheckBox(' Apply Gaussian Filter')

        # Sigma value box, frame, label, and settings
        self.sigmaBox = QtWidgets.QDoubleSpinBox()
        self.sigmaBox.setMaximum(3)
        self.sigmaBox.setDecimals(3)
        self.sigmaBox.setValue(1)
        self.sigmaBox.setMinimum(0.001)

        sigmaFrm = QtWidgets.QFrame()
        sigmaLt = QtWidgets.QGridLayout(sigmaFrm)
        sigmaLt.setContentsMargins(24, 0, 0, 0)
        self.addPair(sigmaLt, 'Sigma: ', self.sigmaBox, 0, 0, 1, 1)

        # Add sub layouts and widgets into frame
        layout.addLayout(colorLt, 0, 0, 1, 1)
        layout.addWidget(self.outlineCheck, 1, 0, 1, 1)
        layout.addWidget(self.filterCheck, 0, 1, 1, 1)
        layout.addWidget(sigmaFrm, 1, 1, 1, 1)

        return frame

    def openColorSelect(self):
        # Open color selection dialog and connect to line color update function
        clrDialog = QtWidgets.QColorDialog(self.maskFrame)
        clrDialog.show()
        clrDialog.colorSelected.connect(self.setMaskColor)

    def setMaskColor(self, color):
        self.maskColor = color.getRgb()[0:3]
        styleSheet = "* { background:" + color.name() + " }"
        self.colorBox.setStyleSheet(styleSheet)
        self.colorBox.show()

    def adjustWindowSize(self):
        # Resize window after plot is made visible
        if not self.gview.isVisible():
            self.gview.setVisible(True)
            self.layout.invalidate()
            size = self.maskFrame.size()
            height, width = size.height(), size.width()
            height = min(height+750, 750)
            width = min(950, width+950)
            self.maskFrame.resize(width, height)

class MaskTool(QtWidgets.QFrame):
    def __init__(self, toolFrame, plotType, parent=None):
        super().__init__(parent)
        self.tool = toolFrame
        self.window = toolFrame.window
        self.plotType = plotType

        # Make a list of plot types that don't have a log color scale
        waveObj = waveanalysis.DynamicWave(self.window)
        waveNames = list(waveObj.defParams.keys())

        self.ui = MaskToolUI()
        self.ui.setupUI(self, toolFrame, plotType, waveNames)

        self.linearColorPlots = ['Coherence', 'Phase']
        self.linearColorPlots += waveObj.plotGroups['Ellipticity']
        self.linearColorPlots += waveObj.plotGroups['Angle']

        # Default plot info
        self.defaultPlotInfo = {
            'Spectra': (None, 'Power', 'nT^2/Hz'),
            'Coherence': ((0, 1), 'Coherence', None),
            'Phase': ((-180, 180), 'Angle', 'Degrees')
        }
        waveObj.numThreads = 1
        for kw in waveObj.defParams.keys():
            self.defaultPlotInfo[kw] = waveObj.defParams[kw]

        # Cross-plot mask support
        self._overrideMask = None
        self._lastMask = None
        self._lastMaskFT = None
        if hasattr(self.ui, 'applyBtn'):
            self.ui.applyBtn.setEnabled(False)

        self.ui.updateBtn.clicked.connect(self.update)
        if hasattr(self.ui, 'applyBtn'):
            try:
                self.ui.applyBtn.clicked.connect(self.apply_mask_to_other)
            except Exception:
                pass


    def getVarInfos(self):
        varInfo = self.ui.toolMaskLt.getVarInfo()
        return varInfo

    def getMaskRanges(self):
        maskRng = self.ui.toolMaskLt.getMaskRanges()
        return maskRng

    def getColorRng(self, grid):
        # Get the range of values to map colors to for each tool type
        if self.plotType == 'Spectra':
            colorRng = self.tool.getGradRange()
            if colorRng is None:
                colorRng = (np.min(grid[grid>0]), np.max(grid[grid>0]))
        elif self.plotType in ['Coherence', 'Phase']:
            if self.plotType == 'Coherence':
                colorRng = (0, 1.0)
            else:
                colorRng = (-180, 180)
        else:
            colorRng = self.tool.getColorRng(self.plotType, grid)
        
        return colorRng

    def update(self):
        self.ui.adjustWindowSize()

        # Get plot info and parameters from main tool
        grid, freqs, times = self.getValueGrid()
        if self._overrideMask is not None and self._overrideMask.shape != grid.shape:
            nF, nT = grid.shape
            m = self._overrideMask
            rF, rT = m.shape
            if rT != nT:
                m = m[:, :nT] if rT > nT else np.concatenate([m, np.repeat(m[:, [-1]], nT - rT, axis=1)], axis=1)
            if rF != nF:
                m = m[:nF, :] if rF > nF else np.concatenate([m, np.repeat(m[[-1], :], nF - rF, axis=0)], axis=0)
            self._overrideMask = m
        logScale = self.tool.getAxisScaling() == 'Logarithmic'
        varInfo = self.tool.getVarInfo()
        colorRng = self.getColorRng(grid)

        # mask creation
        maskRng = self.ui.toolMaskLt.getMaskRanges()
        if self._overrideMask is not None:
            # Use coherence mask (coherence→spectra/wave path)
            maskColor = self.getMaskColor()
            maskOutline = self.ui.outlineCheck.isChecked()
            maskInfo = (self._overrideMask, maskColor, maskOutline)
        else:
            maskInfo = self.createMask(grid, maskRng)

            # If it's a coherence mask, then enable the button
            if self.plotType == 'Coherence':
                alphaMask = maskInfo[0]
                self._lastMask = alphaMask
                self._lastMaskFT = (np.asarray(freqs), np.asarray(times))
                if hasattr(self.ui, 'applyBtn'):
                    self.ui.applyBtn.setEnabled(True)

        plt = self.getPlotItem(grid, freqs, times, logScale, colorRng, maskInfo)
        lbls = self.getLabels(varInfo, logScale)
        self.setupGrid(plt, lbls, times)

        # Save state and add any plotted lines
        self.plt = plt
        for line in self.tool.lineHistory:
            self.addLineToPlot(line)
    
    def addLineToPlot(self, line):
        # Make a copy of the line item and add it to the plot
        pen = line.opts['pen']
        self.plt.plot(line.xData, line.yData, pen=pen)

    def setupGrid(self, plt, lbls, times):
        title, axisLbl, legendLbl = lbls

        spec = plt.getSpecData()[0]
        gradLegend = plt.getGradLegend(logMode=spec.log_color_scale())
        gradLegend.setRange(spec.get_gradient(), spec.get_value_range())
        gradLegend.setBarWidth(38)
        gradLegend.setLabel(legendLbl)

        # Set custom gradient legend tick spacing
        spacing = self.tool.getGradTickSpacing(self.plotType)
        if spacing is not None:
            major, minor = spacing
            gradLegend.setTickSpacing(major, minor)

        # Overlay masked value ranges on the legend
        # Pull from the mask UI settings so the legend reflects them
        alphaMin, alphaMax, alphaOp = self.ui.toolMaskLt.getMaskRanges()
        vmin, vmax = gradLegend.getValueRange()
        logColor = spec.log_color_scale()

        def to_axis_units(v):
            # convert to axis units if the color scale is logarithmic
            return np.log10(v) if (v is not None and logColor) else v

        intervals = []
        # Build intervals in axis units
        amin_ax = to_axis_units(alphaMin)
        amax_ax = to_axis_units(alphaMax)

        if amin_ax is not None and amax_ax is not None:
            if alphaOp == 'AND':
                # [min, max]
                intervals = [(amin_ax, amax_ax)]
            else:
                # Two bars if OR operation: [vmin, max] and [min, vmax]
                intervals = [(vmin, amax_ax), (amin_ax, vmax)]
        elif amin_ax is not None:
            intervals = [(amin_ax, vmax)]
        elif amax_ax is not None:
            intervals = [(vmin, amax_ax)]
        else:
            intervals = []

        # Apply mask color to the legend
        maskColor = self.getMaskColor()  # (r,g,b)
        gradLegend.setLegendMask(intervals, maskColor)

        plt.setTitle(title, size='13pt')
        plt.getAxis('left').setLabel(axisLbl)
    
        # Time info
        timeInfo = self.tool.getTimeInfoLbl((times[0], times[-1]))

        self.ui.glw.clear()
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.ui.glw.addItem(gradLegend, 0, 1, 1, 1)
        self.ui.glw.addItem(legendLbl, 0, 2, 1, 1)
        self.ui.glw.addItem(timeInfo, 1, 0, 1, 3)

    def getLabels(self, varInfo, logScale):
        if self.plotType == 'Spectra':
            lbls = self.tool.getLabels(varInfo, logScale)
        elif self.plotType in ['Coherence', 'Phase']:
            lbls = self.tool.getLabels(self.plotType, varInfo, logScale)
        else:
            lbls = self.tool.getLabels(self.plotType, logScale)
        return lbls

    def getValueGrid(self):
        if self.plotType in ['Coherence', 'Phase']:
            freqs, times, cohGrid, phaGrid = self.tool.lastCalc
            if self.plotType == 'Coherence':
                grid = cohGrid
            else:
                grid = phaGrid
        else:
            times, freqs, grid = self.tool.lastCalc
        
        return grid, freqs, times

    def createMask(self, grid, maskRng):
        # If filter box is checked, apply a Gaussian filter to the grid before
        # the masks are generated
        filtered = self.ui.filterCheck.isChecked()
        if filtered:
            sigma = self.ui.sigmaBox.value()
            grid = ndimage.gaussian_filter(grid, sigma=sigma)

        # Extract parameters
        alphaMin, alphaMax, alphaOp = maskRng

        # Build masks that select everything by default
        alphaMask = np.full(grid.shape, True)

        # Apply mask if min/max value is set for each grid
        if alphaMin is not None:
            alphaMask = np.logical_and(alphaMask, (grid > alphaMin))

        if alphaMax is not None:
            # If both masks are set, apply the binary operation to the masks here
            if alphaMin and alphaOp == 'OR':
                alphaMask = np.logical_or(alphaMask, (grid < alphaMax))
            else:
                alphaMask = np.logical_and(alphaMask, (grid < alphaMax))

        maskOutline = self.ui.outlineCheck.isChecked()

        # If no masks were applied, unmask everything
        if alphaMin is None and alphaMax is None:
            alphaMask = np.logical_not(alphaMask)
            maskOutline = False

        return alphaMask, self.getMaskColor(), maskOutline

    def getMaskColor(self):
        return self.ui.maskColor

    def extendFreqs(self, freqs):
        # Get lower bound for frequencies on plot
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and self.ui.scaleModeBox.currentText() == 'Logarithmic':
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])
        return freqs

    def getPlotItem(self, grid, freqs, times, logScale, colorRng, maskInfo):
        # Determine if color map should interpret the values on a log scale
        if self.plotType in self.linearColorPlots:
            logColorScale = False
        else:
            logColorScale = True

        # Wrapped colors should be used for phase plots
        # A regular spectrogram plot item should be used otherwise
        if self.plotType == 'Phase':
            plt = MagPyPlotItem(self.window.epoch)
        else:
            plt = MagPyPlotItem(self.window.epoch)

        # Get the lower bound for the frequencies and generate the plot
        # from the grid values and mask info
        freqs = self.extendFreqs(freqs)
        plt.createPlot(freqs, grid, times, colorRng, logColorScale, maskInfo=maskInfo)

        return plt
    
    def _remap_mask_to(self, mask_src, freqs_src, times_src, freqs_dst, times_dst):
        # Build index maps
        t_idx = np.clip(np.searchsorted(times_src, times_dst) - 1, 0, len(times_src)-1)
        f_idx = np.clip(np.searchsorted(freqs_src, freqs_dst), 0, len(freqs_src)-1)
        return mask_src[np.ix_(f_idx, t_idx)]
    
    def apply_mask_to_other(self):
        # Only valid if we have a coherence mask cached
        if self.plotType != 'Coherence' or self._lastMask is None or self._lastMaskFT is None:
            return

        target = getattr(self.ui.applyTarget, 'currentText', lambda: '—')()
        if target not in ('Dynamic Spectra', 'Dynamic Wave Analysis'):
            return

        # Clone state (fftInt/shift/bw/scale/detrend), keep the same time selection via window
        state = self.tool.getState()

        freqs_src, times_src = self._lastMaskFT
        alpha_mask_src = self._lastMask

        if target == 'Dynamic Spectra':
            from .dynamicspectra import DynamicSpectra
            spec = DynamicSpectra(self.window)

            # Get the first var A from (A x B)
            try:
                varA, _ = self.tool.getVarInfo()
                state['varInfo'] = varA
            except Exception:
                pass
            state['plotType'] = 'Spectra'
            spec.loadState(state)
            spec.update()

            times_dst, freqs_dst, grid_dst = spec.lastCalc
            nF, nT = grid_dst.shape

            remapped = self._remap_mask_to(alpha_mask_src, freqs_src, times_src,
                                        freqs_dst, times_dst, (nF, nT))

            mt = MaskTool(spec, 'Spectra')
            mt._overrideMask = remapped
            mt.update()
            mt.show()

        else:
            from .waveanalysis import DynamicWave
            wave = DynamicWave(self.window)

            wave_func = self.ui.waveFunc.currentText() if hasattr(self.ui, 'waveFunc') else None
            if not wave_func:
                wave_func = 'Power Spectra Trace'

            state['plotType'] = wave_func
            wave.loadState(state)
            wave.update()

            times_dst, freqs_dst, grid_dst = wave.lastCalc
            nF, nT = grid_dst.shape

            remapped = self._remap_mask_to(alpha_mask_src, freqs_src, times_src,
                                        freqs_dst, times_dst, (nF, nT))

            mt = MaskTool(wave, wave_func if wave_func else 'Wave')
            mt._overrideMask = remapped
            mt.update()
            mt.show()

    
    def _to_numeric_1d(self, arr):
        """Convert times/freqs to a numeric 1D float array for mapping"""
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.datetime64):
            return a.astype('datetime64[ns]').astype(np.int64).astype(np.float64)
        return np.ravel(a).astype(np.float64, copy=False)

    def _index_map_nn(self, src_vals, dst_vals):
        """Map each dst position to the nearest src index using searchsorted"""
        s = self._to_numeric_1d(src_vals)
        d = self._to_numeric_1d(dst_vals)
        n = s.size
        if n == 0:
            return np.zeros_like(d, dtype=int)
        if n == 1:
            return np.zeros_like(d, dtype=int)

        reversed_order = s[0] > s[-1]
        if reversed_order:
            s = s[::-1]

        # Right insertion positions
        r = np.searchsorted(s, d, side='right')
        # Left = r-1
        left  = np.clip(r - 1, 0, n - 1)
        right = np.clip(r,     0, n - 1)

        # Choose the nearer of left or right
        dl = np.abs(d - s[left])
        dr = np.abs(d - s[right])
        idx = np.where(dl <= dr, left, right).astype(int)

        # restore indices to original orientation
        if reversed_order:
            idx = (n - 1) - idx

        np.clip(idx, 0, n - 1, out=idx)
        return idx

    def _remap_mask_to(self, mask_src, freqs_src, times_src, freqs_dst, times_dst, grid_shape):
        """
        Remap boolean mask from (freqs_src, times_src) → (freqs_dst, times_dst)
        using nearest-neighbor on each axis. Returns mask with EXACT shape `grid_shape`.
        """
        nF, nT = int(grid_shape[0]), int(grid_shape[1])

        # Build index maps into mask
        f_idx = self._index_map_nn(freqs_src, np.asarray(freqs_dst)[:nF])
        t_idx = self._index_map_nn(times_src, np.asarray(times_dst)[:nT])

        m = np.asarray(mask_src, dtype=bool)

        if m.ndim != 2:
            raise ValueError("mask_src must be 2D")
        if m.shape != (len(freqs_src), len(times_src)) and m.shape == (len(times_src), len(freqs_src)):
            m = m.T

        f_idx = np.clip(f_idx, 0, m.shape[0] - 1)
        t_idx = np.clip(t_idx, 0, m.shape[1] - 1)

        remap = m[np.ix_(f_idx, t_idx)]

        if remap.shape[1] != nT:
            if remap.shape[1] > nT:
                remap = remap[:, :nT]
            else:
                pad = np.repeat(remap[:, [-1]], nT - remap.shape[1], axis=1)
                remap = np.concatenate([remap, pad], axis=1)
        if remap.shape[0] != nF:
            if remap.shape[0] > nF:
                remap = remap[:nF, :]
            else:
                pad = np.repeat(remap[[-1], :], nF - remap.shape[0], axis=0)
                remap = np.concatenate([remap, pad], axis=0)

        assert remap.shape == (nF, nT), f"Remap shape {remap.shape} != {nF,nT}"
        return remap
