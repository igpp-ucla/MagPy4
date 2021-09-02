

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
import numpy as np
from .plotbase import MagPyPlotItem
from .spectraui import SpectraUI
from .waveanalysis import WaveAnalysis
import os
from bisect import bisect_left, bisect_right
from scipy.interpolate import CubicSpline
from .spectraalg import SpectraCalc

class ColorPlotTitle(pg.LabelItem):
    ''' LabelItem with horizontally stacked labels in given colors '''
    def __init__(self, labels, colors, sep=', ', *args, **kwargs):
        self.labels = labels
        self.colors = colors
        self.sep = sep

        super().__init__('', size='11pt')

        self.updateText()

    def getColors(self):
        return self.colors

    def setColors(self, colors):
        ''' Set new colors for the sublabels '''
        self.colors = colors
        self.updateText()
    
    def getLabels(self):
        return self.labels

    def setLabels(self, labels):
        ''' Set new sublabels '''
        self.labels = labels
        self.updateText()

    def updateText(self):
        ''' Update HTML with current labels and colors '''
        substrs = []
        for label, color in zip(self.labels, self.colors):
            html = self.htmlColor(label, color)
            substrs.append(html)
        
        self.item.setHtml(self.sep.join(substrs))

    def htmlColor(self, txt, color):
        ''' Formats text with color in HTML string '''
        return f'<span style="color:{color}">{txt}</span>'

class SpectraPlot(MagPyPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for ax in ['left', 'bottom']:
            self.getAxis(ax).enableAutoSIPrefix(False)

    def setTitleObject(self, titleObj):
        ''' Replaces title label object '''
        self.layout.removeItem(self.titleLabel)
        self.layout.addItem(titleObj, 0, 1)
        self.titleLabel = titleObj

    def getTitleObject(self):
        ''' Returns title label object '''
        return self.titleLabel

class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self, window)
        self.grid_map = {
            'spectra' : self.ui.grid,
            'coherence' : self.ui.cohGrid,
            'phase' : self.ui.phaGrid,
            'sum' : self.ui.sumGrid
        }
        
        self.ui.updateButton.clicked.connect(self.update)
        self.ui.bandWidthSpinBox.valueChanged.connect(self.update)
        self.ui.separateTracesCheckBox.stateChanged.connect(self.toggleSepTraces)
        self.ui.aspectLockedCheckBox.stateChanged.connect(self.toggleLockAspect)
        self.ui.waveanalysisButton.clicked.connect(self.openWaveAnalysis)
        self.ui.logModeCheckBox.toggled.connect(self.toggleLogScale)
        self.ui.unitRatioCheckbox.stateChanged.connect(self.toggleUnitRatio)

        # Cached data
        self.currRange = None # Current index range
        self.selectInfo = None # Currently selected plots
        self.dataState = None # If data are all the same length
        self.powers = {}
        self.ffts = {}
        self.data = {}
        self.pens = {}
        self.maxN = 0
        self.wasClosed = False

        # Stored plot items
        self.plots = {}

        # Wave analysis and plot appearance objects
        self.waveanalysis = None
        self.plotAppr = None

    def getRange(self):
        return self.ui.timeEdit.getRange()
    
    def checkCache(self):
        ''' Checks if cache and previous plots need to be cleared '''
        currRange = self.getRange()
        clear = not (self.currRange == currRange)
        self.currRange = currRange
        return clear
    
    def checkSelection(self, selectInfo, clear):
        ''' Checks if selection has changed when replotting '''
        res = (self.selectInfo == selectInfo)
        if not res or clear:
            self.dataState = self.getDataState(selectInfo)
        self.selectInfo = selectInfo
        return res

    def getDataState(self, selectInfo):
        # Extract all plot variables
        dstrs = []
        for plot_dstrs, pens in selectInfo:
            dstrs.extend(plot_dstrs)
        
        # Check time lengths of dstrs
        time_indices = set()
        for dstr, en in dstrs:
            index = self.window.getTimeIndex(dstr, en)
            time_indices.add(index)
        
        # Check if times do not match
        if len(time_indices) > 1:
            self.interpData(dstrs)
            return True
        else:
            return False

    def interpData(self, dstrs):
        ''' Finds the largest time array for the given plot
            variables and interpolates all variable along
            that time array and stores it in dictionary
        '''
        # Get largest time range
        max_tlen = 0
        max_times = []
        for dstr, en in dstrs:
            # Get and clip time range for each variable within
            # the selected interval
            times = self.window.getTimes(dstr, en)[0]
            sI, eI = self.getIndices(dstr, en)
            times = times[sI:eI]

            # Check if this is the longest time array
            if len(times) > max_tlen:
                max_tlen = len(times)
                max_times = times
        
        # Interpolate data along max times and save in cache
        interpTimes = max_times
        for dstr, en in dstrs:
            times = self.window.getTimes(dstr, en)[0]
            data = self.window.getData(dstr, en)
            cs = CubicSpline(times, data, extrapolate=False)
            interpData = cs(interpTimes)

            self.data[(dstr, en)] = interpData

    def update(self):
        ''' Update all plots '''
        # Clear cache if indices have changed
        clear = self.checkCache()
        if clear:
            self.data = {}
            self.maxN = 0
            self.ffts = {}

        # Re-initialize plots if selected plot variables have changed
        info = self.window.getSelectedPlotInfo()
        if not self.checkSelection(info, clear) or clear:
            split_trace = self.ui.separateTracesCheckBox.isChecked()
            self.initPlots(split_trace)

        # Get parameters
        params = self.getParams()

        # Update each plot group
        self.updateSpectra(info, params)
        self.updateCohPha(params)
        self.updateSumPowers(params)

        # Update y-axis ranges
        self.updateRanges()

    def getParams(self):
        ''' Get parameters from UI '''
        # Get bandwidth
        bw = self.ui.bandWidthSpinBox.value()

        # Get coherence/phase pair
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()

        # Get sum of powers bool and vector
        sum_powers = self.ui.combinedFrame.isChecked()
        sum_vec = []
        if sum_powers:
            sum_vec = [box.currentText() for box in self.ui.axisBoxes]

        # Get separate traces bool
        split_trace = self.ui.separateTracesCheckBox.isChecked()

        # Assemble parameters dictionary
        params = {
            'bandwidth' : bw,
            'pair' : (c0, c1),
            'sum_powers' : (sum_powers),
            'sum_vec' : (sum_vec),
            'split_trace' : split_trace,
        }

        return params

    def initPlots(self, split_trace):
        ''' Initialize all plot grids
            - split_trace indicates whether spectra plots
              should have a new plot for each variable
        '''
        # Clear previous plots
        grids = [self.ui.grid, self.ui.cohGrid, self.ui.phaGrid, 
            self.ui.sumGrid]
        for grid in grids:
            grid.clear()

        # Get previous plot info        
        plotInfo = self.window.getSelectedPlotInfo()

        # Get each set of plots
        self.plots['spectra'] = self.setupSpectraPlots(plotInfo, split_trace)
        coh, pha = self.setupCohPhaPlots()
        self.plots['coherence'] = [coh]
        self.plots['phase'] = [pha]
        self.plots['sum'] = []

        # Add plots to respective grids
        groups = ['spectra', 'coherence', 'phase']
        for grp, grid in zip(groups, grids):
            for plot in self.plots[grp]:
                # Add plot to grid
                grid.addItem(plot)

                # Create a plot appearance action and add to plot
                act = QtWidgets.QAction(self)
                act.setText('Change Plot Appearance...')

        # Update y scaling
        self.toggleLogScale(self.ui.logModeCheckBox.isChecked())

        # Enable grid tracking
        views = [self.ui.gview, self.ui.cohView, self.ui.phaView, self.ui.combView]
        for grid, view in zip(grids, views):
            grid.enableTracking(True, viewWidget=view)
    
    def splitTraceInfo(self, plotInfo, split_trace):
        ''' Split plot variable and pen info, individually
            if split_trace is True
        '''
        dstrs = []
        pens = {'spectra':[]}
        for plot in plotInfo:
            plot_dstrs, plot_pens = plot
            if split_trace:
                dstrs.extend([[dstr] for dstr in plot_dstrs])
                pen_set = [[pen] for pen in plot_pens]
            else:
                dstrs.append(plot_dstrs)
                pen_set = [plot_pens]
            
            pens['spectra'].extend(pen_set)

        return dstrs, pens

    def setupSpectraPlots(self, plotInfo, split_trace=False):
        ''' Initialize spectra plots '''
        plot_dstrs, spec_pens = self.splitTraceInfo(plotInfo, split_trace)
        self.pens['spectra'] = spec_pens['spectra']

        # Iterate over each plot variable list to create a plot item for it
        plots = []
        for dstrs, pens in zip(plot_dstrs, self.pens['spectra']):
            # Create plot item
            plot = SpectraPlot()
            plots.append(plot)

            # Set plot item title
            var_labels = [self.window.getLabel(dstr, en) for (dstr, en) in dstrs]
            colors = [pen for pen in pens]
            label = ColorPlotTitle(var_labels, colors)
            plot.setTitleObject(label)

            # Set axis labels
            plot.getAxis('left').setLabel('Power (nT<sup>2</sup> Hz<sup>-1</sup>)')
            plot.getAxis('bottom').setLabel('Frequency (Hz)')
            plot.hideAxis('right')
            plot.hideAxis('top')
            plot.setLogMode(y=True)

            # Disable auto range
            plot.enableAutoRange(y=False, x=True)

        # Link plot y axes
        for i in range(1, len(plots)):
            plot_a = plots[i-1]
            plot_b = plots[i]
            plot_b.setYLink(plot_a.getViewBox())
        
        # Add in time label
        self.ui.labelLayout.clear()
        start, end = self.getRange()
        start = start.strftime('%Y %b %d %H:%M:%S.%f')[:-3]
        end = end.strftime('%Y %b %d %H:%M:%S.%f')[:-3]
        label = f'Time: {start} to {end}'
        self.ui.labelLayout.addLabel(label)

        return plots

    def setupCohPhaPlots(self):
        ''' Initialize coherence and phase plots '''
        # Attributes for each plot
        labels = ['Coherence', 'Phase']
        units = ['Coherence', 'Angle (deg)']
        spacing = [[(0.2, 0), (0.1, 0)], [(30, 0), (15, 0)]]
        ranges = [(0., 1.), (-200, 200)]

        # Iterate over plot items and create each plot item
        plots = []
        for label, unit, spacers, rng in zip(labels, units, spacing, ranges):
            plot = MagPyPlotItem()
            plots.append(plot)

            # Set attributes
            plot.setTitle(label)
            left_axis = plot.getAxis('left')
            left_axis.setLabel(unit)
            left_axis.setTickSpacing(levels=spacers)
            plot.getAxis('bottom').setLabel('Frequency (Hz)')

            # Set plot ranges
            vb = plot.getViewBox()
            min_val, max_val = rng
            vb.setLimits(yMin=min_val, yMax=max_val)

        return plots

    def setupSumPlots(self):
        ''' Set up sum of powers spectra plots '''
        # Get general pens from main window
        pens = self.window.pens[:3]

        # Set up plot attributes
        labels = [['Px + Py + Pz'], ['Pt'], ['|Px + Py + Pz - Pt|'], 
            ['|Px + Py + Pz - Pt|', 'Pt']]
        trace_pens = [[pens[0]], [pens[1]], [pens[2]], [pens[2], pens[1]]]

        left_label = 'Power (nT<sup>2</sup> Hz<sup>-1</sup>)'
        btm_label = 'Frequency (Hz)'

        # Create plots
        plots = []
        self.pens['sum'] = {}
        for label_set, pen_set in zip(labels, trace_pens):
            plot = SpectraPlot()
            plots.append(plot)

            # Set plot labels
            plot.setLabel('left', left_label)
            plot.setLabel('bottom', btm_label)

            # Plot title
            colors = [pen.color().name() for pen in pen_set]
            title = ColorPlotTitle(label_set, colors)
            plot.setTitleObject(title)

            # Log scale for power
            plot.setLogMode(y=True)

            # Save plot pens
            for label, pen in zip(label_set, pen_set):
                self.pens['sum'][label] = pen

        return plots

    def updateSpectra(self, info=None, params={}):
        ''' Calculate and re-plot spectra plots '''
        if info is None:
            info = self.window.getSelectedPlotInfo()

        split_trace = params.get('split_trace')
        plot_dstrs, pens = self.splitTraceInfo(info, split_trace)

        # Iterate over each plot variable list, plot items, and pens
        freq = None
        datas = []
        plots = self.plots['spectra']
        plot_pens = self.pens['spectra']
        for dstrs, plot, pens in zip(plot_dstrs, plots, plot_pens):
            # Clear plot
            plot.clear()
            for (dstr, en), pen in zip(dstrs, pens):
                # Get variable data
                data = self.getData(dstr, en)
                res = self.window.getTimes(dstr, en)[-1]

                # Get frequencies and n
                if freq is None:
                    n = len(data)
                    bw = params.get('bandwidth')
                    freq = SpectraCalc.calc_freq(bw, n, res)
                    self.maxN = max(self.maxN, n)

                # Calculate power
                power = SpectraCalc.calc_power(data, bw, res)
                datas.append(power)

                # Plot trace
                label = self.window.getLabel(dstr, en)
                plot.plot(freq, power, pen=pen, name=label)
        
        return freq, datas
    
    def updateCohPha(self, params={}):
        ''' Calculate and replot coherence/phase plots '''
        # Clear previous plots
        self.plots['coherence'][0].clear()
        self.plots['phase'][0].clear()

        # Get variable pair
        vara, varb = params.get('pair')
        
        # Calculate FFT
        fft0 = self.getfft(vara, 0)
        fft1 = self.getfft(varb, 0)

        # Calculate coherence and phase and frequency
        bw = params.get('bandwidth')
        res = self.window.getTimes(vara, self.window.currentEdit)[-1]
        n = len(self.getData(varb, 0))
        freq = SpectraCalc.calc_freq(bw, n, res)
        coh, pha = SpectraCalc.calc_coh_pha_fft(fft0, fft1, bw, res, n)

        # Get pen and plot
        pen = self.window.pens[0]
        self.plots['coherence'][0].plot(freq, coh, pen=pen)
        self.plots['phase'][0].plot(freq, pha, pen=pen)
        self.pens['coherence'] = [[pen]]
        self.pens['phase'] = [[pen]]

        return (freq, [coh, pha])

    def updateSumPowers(self, params):
        ''' Calculate and plot sum of power spectra plots '''
        # Check if this needs to be calculated
        val = params.get('sum_powers')
        if not val:
            self.ui.tabs.setTabEnabled(3, False)
            return
        
        # If not enabled, enable tab and set up plots
        if not self.ui.tabs.isTabEnabled(3) or len(self.ui.sumGrid.items) < 1:
            self.ui.sumGrid.clear()
            self.plots['sum'] = self.setupSumPlots()
            for plot in self.plots['sum']:
                self.ui.sumGrid.addItem(plot)
                plot.setLogMode(x=self.ui.logModeCheckBox.isChecked())

            self.ui.sumGrid.enableTracking(True, viewWidget=self.ui.combView)
            self.ui.tabs.setTabEnabled(3, True)

        # Calculate variables
        sum_powers, pt, sum_minus, freq = self.calculateSumOfPowers(params)

        # Plot traces
        labels = [['Px + Py + Pz'], ['Pt'], ['|Px + Py + Pz - Pt|'], 
            ['|Px + Py + Pz - Pt|', 'Pt']]
        groups = [[sum_powers], [pt], [sum_minus], [sum_minus, pt]]
        plots = self.plots['sum']

        for plot, label_set, values in zip(plots, labels, groups):
            plot.clear()
            for label, value in zip(label_set, values):
                plot.plot(freq, value, pen=self.pens['sum'][label], name=label)
        
        return freq, [sum_powers, pt, sum_minus]
    
    def calculateSumOfPowers(self, params):
        # Extract parameters
        bw = params.get('bandwidth')
        vec = params.get('sum_vec')
        en = self.window.currentEdit
        sI, eI = self.getIndices(vec[0], en)
        n = abs(eI - sI)
        res = self.window.getTimes(vec[0], en)[-1]

        # Compute Btotal and its fft if not previously computed
        key = '_'.join(vec + ['mag'])
        if key not in self.ffts:
            data = np.vstack([self.window.getData(dstr, en)[sI:eI] for dstr in vec])
            mag = np.sqrt(np.sum(data ** 2, axis=0))
            fft = SpectraCalc.calc_fft(mag)
            self.ffts[key] = fft

        mag_fft = self.ffts[key]

        # Calculate powers and sums
        bffts = [self.getfft(dstr, en) for dstr in vec]

        px, py, pz = [SpectraCalc.calc_power_fft(fft, bw, res, n) for fft in bffts]
        pt = SpectraCalc.calc_power_fft(mag_fft, bw, res, n)

        sum_powers = px + py + pz
        sum_minus = abs(sum_powers - pt)

        # Compute frequencies
        freqs = SpectraCalc.calc_freq(bw, n, res)
        
        return sum_powers, pt, sum_minus, freqs

    def updateRanges(self):
        ''' Updates y-ranges for all plots '''
        # Iterate over each plot group
        for grp in self.plots:
            lower, upper = None, None
            # Find the plot data items for each plot in group
            # and find the min and max y values in the plot
            # traces
            for plot in self.plots[grp]:
                pdis = plot.listDataItems()
                for pdi in pdis:
                    ymin, ymax = pdi.dataBounds(ax=1)
                    if lower is None:
                        lower = ymin
                        upper = ymax
                    else:
                        lower = min(ymin, lower)
                        upper = max(ymax, upper)
            
            # Set y ranges for all groups to extreme values and add padding
            for plot in self.plots[grp]:
                plot.setYRange(lower, upper, 0.01)

    def getData(self, dstr, en):
        ''' Returns the data for the given variable,
            interpolated if there are multiple time
            arrays for the selected set of variables
        '''
        # Return interpolated data if saved
        if (dstr, en) in self.data:
            return self.data[(dstr, en)]

        # Get data and start/end indices
        data = self.window.getData(dstr, en)
        sI, eI = self.getIndices(dstr, en)

        return data[sI:eI]

    def getIndices(self, dstr, en):
        ''' Returns start/end indices for variable '''
        # Get start/end datetimes
        start, end = self.getRange()

        # Convert to ticks
        start = self.window.getTickFromDateTime(start)
        end = self.window.getTickFromDateTime(end)

        # Find indices in time array
        times = self.window.getTimes(dstr, en)[0]
        sI = bisect_left(times, start)
        eI = bisect_left(times, end)

        return (sI, eI)
    
    def getfft(self, dstr, en):
        ''' Return FFT for data, calculate if not
            cached
        '''
        # Calculate fft if not in cache
        if (dstr, en) not in self.ffts:
            data = self.getData(dstr, en)
            fft = SpectraCalc.calc_fft(data)
            self.ffts[(dstr, en)] = fft

        return self.ffts[(dstr, en)]

    def closeWaveAnalysis(self):
        ''' Closes wave analysis window '''
        if self.waveanalysis:
            self.waveanalysis.close()
            self.waveanalysis = None

    def openWaveAnalysis(self):
        ''' Opens wave analysis window '''
        self.closeWaveAnalysis()
        self.waveanalysis = WaveAnalysis(self, self.window)
        self.waveanalysis.show()

    def closeEvent(self, event):
        if self.wasClosed:
            return

        self.closeWaveAnalysis()
        self.wasClosed = True
        self.window.endGeneralSelect()

    def toggleSepTraces(self, val):
        ''' Toggles whether each variable should have its own plot
            or not
        '''
        # Clear and rebuild spectra plots
        self.ui.grid.clear()
        info = self.window.getSelectedPlotInfo()
        self.plots['spectra'] = self.setupSpectraPlots(info, val)
        for plot in self.plots['spectra']:
            self.ui.grid.addItem(plot)
        self.ui.grid.enableTracking(True, viewWidget=self.ui.gview)
        self.toggleLogScale(self.ui.logModeCheckBox.isChecked())
        self.update()

    def toggleLogScale(self, logScale):
        ''' Toggles log y-axis scaling '''
        for grp in self.plots:
            for plot in self.plots[grp]:
                plot.setLogMode(x=logScale)

    def toggleUnitRatio(self, val):
        ''' Toggles 1:1 aspect ratio '''
        ratio = 1.0 if val else None
        for grp in ['spectra', 'sum']:
            for plot in self.plots[grp]:
                plot.setAspectLocked(val, ratio=ratio)
        self.updateRanges()

    def toggleLockAspect(self, val):
        ''' Toggles a locked aspect ratio '''
        for grp in ['spectra', 'sum']:
            for plot in self.plots[grp]:
                plot.setAspectLocked(val, ratio=None)

    def updateTitleColors(self, plot_type, plot_info):
        # Extract color and plot trace info
        changed_plot, name, (old_color, new_color) = plot_info

        # Find corresponding plot in grid
        grid = self.grid_map[plot_type]
        plots = grid.get_plots()
        index = None
        for i in range(len(plots)):
            if plots[i] == changed_plot:
                index = i
                break

        # Do not change stored title color if has no color set
        if not isinstance(changed_plot, SpectraPlot):
            self.pens[plot_type][index][0] = new_color.name()
            return

        # Find corresponding trace in plot
        if index is not None:
            label_index = None
            labels = changed_plot.getTitleObject().getLabels()
            i = 0
            for label in labels:
                if label == name:
                    label_index = i
                    break
                i += 1
            self.pens[plot_type][index][label_index] = new_color.name()

class SpectraTests():
    def __init__(self, window):
        self.window = window
        self.file_folder = 'test_files'
        self.specFiles = ['spec_bx.csv', 'spec_by.csv', 'spec_bz.csv']
        self.cohFiles = ['coh_bx_by.csv']
        self.phaseFiles = ['phase_bx_by.csv']
        self.sumFiles = ['sum_spec.csv', 'pt_spec.csv', 'sum_minus.csv']

        self.window.startTool('Spectra')
        self.tool = self.window.tools['Spectra']
        self.window.currSelect.setPostFunc(self.runTests)
        self.window.show()
        self.window.autoSelectRange()

    def runTests(self):
        self.tool.ui.combinedFrame.setChecked(True)
        self.tool.update()
        params = self.tool.getParams()
        self.test_spectra(params)
        self.test_coh_pha(params)
        self.test_sum_powers(params)

    def test_spectra(self, params):
        ''' Test spectra plot results '''
        freq, datas = self.tool.updateSpectra(info=None, params=params)
        self.test_datas(freq, datas, self.specFiles, True)

    def test_coh_pha(self, params):
        ''' Test coherence and phase calculations '''
        freq, data = self.tool.updateCohPha(params)
        files = self.cohFiles + self.phaseFiles
        self.test_datas(freq, data, files, False)

    def test_sum_powers(self, params):
        ''' Test sum of powers calculations '''
        freq, datas = self.tool.updateSumPowers(params)
        self.test_datas(freq, datas, self.sumFiles, True)

    def test_datas(self, freq, datas, files, log_y):
        ''' Iterate over datas and files to see if
            the y values are the same and that the x values
            match the calculated frequencies

            log_y specifies whether y-values should be log-scaled
        '''
        axes = np.arange(len(files))

        for axis, file in zip(axes, files):
            file = os.path.join(self.file_folder, file)
            test_data = np.loadtxt(file, delimiter=',', skiprows=1, usecols=[0,1])
            
            x0 = 10 ** test_data[:,0]
            if log_y:
                y0 = 10 ** test_data[:,1]
            else:
                y0 = test_data[:,1]

            calc_data = np.array(datas[axis])
            calc_freq = np.array(freq)

            assert(np.allclose(x0, calc_freq))
            assert(np.allclose(y0, calc_data))