import numpy as np
from ..dynBase import SpecData, DynamicAnalysisTool
import matplotlib.pyplot as plt
from ..specAlg import SpectraCalc, WaveCalc
import matplotlib
from datetime import datetime, timedelta
import functools
from scipy import signal as scipysig
from .constants import spec_infos

def _spec_wrapper(calc_func, times, sigs, fftint, fftshift, bw, detrend, flag, res):
    ''' Runs the main spectrogram algorithm, using calc_func
        to calculate the values for each column
    '''
    # Stack signals
    if len(sigs) > 1:
        sigstack = np.vstack(sigs)
        n = sigstack.shape[1]
    else:
        sigstack = sigs[0]
        n = len(sigstack)

    # Calculate any unknown parameters
    tool = DynamicAnalysisTool()
    if fftint is None:
        fftint = tool.guess_fft_param(n)
    if fftshift is None:
        fftshift = int(fftint / 4)

    # Check parameter validity
    if fftint < fftshift:
        raise Exception('Interval must be > shift')

    if bw % 2 == 0:
        raise Exception('Bandwidth must be odd')

    # Convert flag values to nan's
    if not np.isnan(flag):
        sigstack[sigstack >= flag] = np.nan

    # Convert times to numpy array
    times = np.array(times, dtype='datetime64[us]')

    # Calculate median resolution from data
    if res is None:
        diffs = np.diff(times)
        diffs = [td.item().total_seconds() for td in diffs]
        res = np.median(diffs)

    # Split time and data into segments
    time_segs, data_segs = tool.splitDataSegments(times, sigstack, fftint, fftshift, onedim=len(sigs) == 1)
    x = np.array(time_segs)

    # Detrend segments if necessary
    if detrend:
        axis = 0 if len(sigstack.shape) < 2 else 1
        data_segs = [scipysig.detrend(seg, axis=axis) for seg in data_segs]

    # Compute grid values
    grid = []
    for seg in data_segs:
        spectra = calc_func(seg, bw, res)
        grid.append(spectra)
    grid = np.array(grid).T

    # Compute frequency list
    y = SpectraCalc.calc_freq(bw, fftint, res)

    # Assemble parameters info
    params = {
        'fftshift' : fftshift,
        'fftint' : fftint,
        'bw' : bw,
        'detrend' : detrend,
    }

    return x, y, grid, params

def _spec_data_wrapper(key, *args, **kwargs):
    # Calls _spec_wrapper and sets SpecData values
    # according to keyword's corresponding values in
    # spec_infos
    x, y, grid, params = _spec_wrapper(*args, **kwargs)
    key_kwargs = spec_infos[key]
    key_kwargs['y_label'] = 'Frequency (Hz)'
    key_kwargs['log_color'] = (key_kwargs['color_rng'] is None)
    key_kwargs['log_y'] = True
    spec = SpecData(y, x, grid, **key_kwargs)

    # Store parameters
    spec.params().update(params)

    return spec

def _coh_wrapper(sigs, bw, res):
    return SpectraCalc.calc_coh_pha(sigs, bw, res)[0]

def _phase_wrapper(sigs, bw, res):
    return SpectraCalc.calc_coh_pha(sigs, bw, res)[1]

def _wave_wrapper(f, sigs, bw, res):
    n = len(sigs[0])
    fftparams = {
        'bandwidth' : bw,
        'resolution' : res,
        'num_points' : n
    }
    return f(sigs, fftparams)

def spectrogram(times, sig, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic spectrogram for a signal

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig : array-like
                Signal to compute power spectra values for
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = SpectraCalc.calc_power
    spec = _spec_data_wrapper('spec', calc_func, times, [sig], fftint, 
        fftshift, bw, detrend, flag, res)
    return spec

def coherence_spec(times, sig1, sig2, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic coherence spectrogram for a pair of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2 : array-like
                Signals to compute coherence values for
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = _coh_wrapper
    spec = _spec_data_wrapper('coh', calc_func, times, [sig1, sig2], fftint, 
        fftshift, bw, detrend, flag, res)
    return spec

def phase_spec(times, sig1, sig2, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic phase spectrogram for a pair of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2 : array-like
                Signals to compute phase values for
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = _phase_wrapper
    spec = _spec_data_wrapper('pha', calc_func, times, [sig1, sig2], fftint, 
        fftshift, bw, detrend, flag, res)
    spec.cmap = 'twilight'
    return spec

def ellipticity_spec(times, sig1, sig2, sig3, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic ellipticity spectrogram for a set of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2,3 : array-like
                Signals to compute ellipticity values for (bx, by, bz)
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = functools.partial(_wave_wrapper, WaveCalc.calc_ellip)
    spec = _spec_data_wrapper('ellip', calc_func, times, [sig1, sig2, sig3], fftint,
        fftshift, bw, detrend, flag, res)
    return spec

def prop_angle_spec(times, sig1, sig2, sig3, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic propagation angle spectrogram for a set of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2,3 : array-like
                Signals to compute propagation angle for (bx, by, bz)
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = functools.partial(_wave_wrapper, WaveCalc.calc_prop_angle)
    spec = _spec_data_wrapper('propangle', calc_func, times, [sig1, sig2, sig3], fftint,
        fftshift, bw, detrend, flag, res)
    return spec

def power_trace_spec(times, sig1, sig2, sig3, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic power spectra trace (Px + Py + Pz) for 
        a set of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2,3 : array-like
                Signal to compute power spectra trace values for (bx, by, bz)
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = SpectraCalc.calc_sum_powers
    spec = _spec_data_wrapper('powertrace', calc_func, times, [sig1, sig2, sig3], fftint,
        fftshift, bw, detrend, flag, res)
    return spec

def compr_power_spec(times, sig1, sig2, sig3, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic compressional power (Pt) spectrogram for a
        set of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2,3 : array-like
                Signals to compute compressional power for (bx, by, bz)
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = SpectraCalc.calc_compress_power
    spec = _spec_data_wrapper('comppower', calc_func, times, [sig1, sig2, sig3], fftint,
        fftshift, bw, detrend, flag, res)
    return spec

def tranv_power_spec(times, sig1, sig2, sig3, fftint=None, fftshift=None, bw=3,
        detrend=False, flag=np.nan, res=None):
    '''
        Calculates the dynamic tranverse power (Px + Py + Pz - Pt) spectrogram
        for a set of signals

        Parameters
        ----------
        times : array-like
                Time array for data in datetime objects
        sig1,2,3 : array-like
                Signals to compute tranverse power values for (bx, by, bz)
        fftint : int, optional
                 FFT interval, number of points to use per FFT
                 Default is None, which will compute an appropriate value
        fftint : int, optional
                 FFT shift, number of points to shift each section by
                 Default is None, which will compute an appropriate value
        bw : int, optional
             Number of frequency bands to average over (must be odd!)
             Default value is 3
        detrend : boolean, optional
                  Specifies whether to detrend each data slice before computing the 
                  fast fourier transform;
                  Default value is False
        flag : float or np.nan, optional
               Specifies the error flag value to use
               Default value is np.nan
        res : float, optional
               Specifies the resolution of the data in seconds
               Default value is None, which will compute an estimated value

        Returns
        ---------
        SpecData instance
            Contains the calculated spectrogram x, y, and z values and
            any other necessary parameters for plotting
    '''
    calc_func = SpectraCalc.calc_tranv_power
    spec = _spec_data_wrapper('tranvpower', calc_func, times, [sig1, sig2, sig3], fftint,
        fftshift, bw, detrend, flag, res)
    return spec

def _log_formatter(pos, val):
    if int(pos) == float(pos):
        pos = int(pos)
    else:
        pos = np.round(pos, 3)
    return f'$10^{{{pos}}}$'

def create_spec_object(x, y, z):
    '''
        Generates a SpecData object from the given bins 
        and grid values

        Parameters
        ----------
        x : array-like (m,)
            List of datetimes representing x-bins
        y : array-like (n,)
            List of floats representing the y-bins
        z : array-like (m-1, n-1,)
            Grid of floats representing the values
            in the grid
    '''
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return SpecData(y, x, z)

def plot_spec_data(spec, figsize=(9, 6), title=None, logy=None,
                cmap=None, rng=None, logcolor=None, smooth=False,
                fig_ax=None):
    ''' 
        Plots a SpecData object and onto a matplotlib Figure

        Parameters
        ----------
        spec : SpecData object
               Data to plot
        figsize : tuple of floats, optional
                  Indicates the fig size (width, height) in inches
                  Default is (9, 6)
        title : str, optional
                Title to use for figure
                Default is None
        logy : boolean, optional
               Indicates whether the y-axis should be on a log-scale or not;
                Default value is None, which will use the spec's value
        cmap : ColorMap, optional
               matplotlib colormap to use for colorbar and grid mapping
               Default value is None
        rng : Tuple of floats, optional
              Specifies a range of values to limit color range to;
              Default is None, which will use the full range of values in
                the data
        logcolor : boolean or None, optional
                    Specifies whether grid values should be mapped to log10
                    values before mapping to color values;
                    Default value is None, which will use the spec's value
        smooth : boolean or None, optional
                    Indicates whether to smooth the blocks in the image
                    Defaults to False
        fig_ax : tuple (matplotlib Figure, matplotlib Axes), optional
                    If this is not None, the plot will be generated on the
                    axis item in the given figure instead of a new figure

        Returns
        ---------
        A tuple - (Figure object, Axes object, Colorbar object)
    '''
    # Get any missing parameters from SpecData
    if title is None:
        title = spec.get_name()

    # Create figure and set title and size
    if fig_ax is None:
        fig, ax = plt.subplots(nrows=1)
        fig.suptitle(title)
        fig.canvas.set_window_title(title)
        fig.set_size_inches(*figsize)
    else:
        fig, ax = fig_ax

    # Get color range and settings
    logcolor = spec.log_color_scale() if logcolor is None else logcolor
    if rng is not None:
        vmin, vmax = rng
        if logcolor:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)
    else:
        vmin, vmax = None, None

    # Extract data values
    x, y, z = spec.values()

    # Adjust points if smoothing
    if smooth:
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        x = x[:-1] + (x_diff / 2)
        y = y[:-1] + (y_diff / 2)

    # Map values to log scale if specified
    if logcolor:
        z = np.array(z)
        mask = (z <= 0)
        z[z <= 0] = 1
        z = np.log10(z)
        z[mask] = np.nan

    # Plot grid colors
    cmap = spec.cmap if cmap is None else cmap
    shading = 'flat' if not smooth else 'gouraud'
    img = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)

    # Set up color bar
    bar = fig.colorbar(img, ax=ax, pad=0.03, aspect=17)
    if logcolor:
        bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_log_formatter))

    # Set labels for y axis and colorbar
    ylbl, lgnd_lbl = spec.get_labels()
    ax.set_ylabel(ylbl)
    bar.set_label(lgnd_lbl)

    # Scale axis if necessary
    logy = spec.log_y_scale() if logy is None else logy
    if logy:
        ax.set_yscale('log')

    # Show top and right ticks
    ax.tick_params(top=True, right=True)
    ax.tick_params(length=4)

    # Additional adjusments to figure if fig_ax not passed
    if fig_ax is None:
        # Add 'Time Range' text
        fmt = '%Y %b %d %H:%M:%S.%f'
        start_str = x[0].item().strftime(fmt)
        end_str = x[-1].item().strftime(fmt)
        time_lbl = f'Time Range: {start_str} to {end_str}'
        fig.text(0.08, 0.015, time_lbl)

        # Adjust figure layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.subplots_adjust(top=0.93)

    return fig, ax, bar