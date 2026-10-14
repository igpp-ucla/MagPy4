''' Dynamic wave analysis: a wave parameter computed over a sliding FFT window
    and drawn as a time/frequency spectrogram.

    The computation mirrors DynamicWave.calcGrid in tools/waveanalysis.py,
    run serially so it needs no main window or worker processes.
'''
import numpy as np
from scipy import signal
from fflib import ff_time

from ...alg.spectraalg import SpectraCalc, WaveCalc
from ...tools.dynbase import DynamicAnalysisTool
from ..prompt import CLIError
from ..plotting import render_spectrogram
from . import Tool, Param, register, IMAGE_FORMATS

# slug -> (name in the GUI, compute function, method argument, default color
# range or None for min/max of the data, colorbar label, units, title)
PLOT_TYPES = {
    'azimuth': ('Azimuth Angle', WaveCalc.calc_azimuth_angle, None,
        (-90, 90), 'Azimuth Angle', 'Degrees', 'Azimuth Angle'),
    'ellipticity-means': ('Ellipticity (Means)', WaveCalc.calc_ellip, 'means',
        (-1.0, 1.0), 'Ellipticity', None, 'Ellipticity (Means)'),
    'ellipticity-svd': ('Ellipticity (SVD)', WaveCalc.calc_ellip, 'svd',
        (-1.0, 1.0), 'Ellipticity', None, 'Ellipticity (SVD)'),
    'ellipticity-bw': ('Ellipticity (Born-Wolf)', WaveCalc.calc_ellip, 'bw',
        (0, 1.0), 'Ellipticity', None, 'Ellipticity (Born-Wolf)'),
    'propangle-means': ('Propagation Angle (Means)', WaveCalc.calc_prop_angle,
        'means', (0, 90), 'Angle', 'Degrees', 'Propagation Angle (Means)'),
    'propangle-svd': ('Propagation Angle (SVD)', WaveCalc.calc_prop_angle,
        'svd', (0, 90), 'Angle', 'Degrees', 'Propagation Angle (SVD)'),
    'propangle-minvar': ('Propagation Angle (Min Var)', WaveCalc.calc_prop_angle,
        'minvar', (0, 90), 'Angle', 'Degrees', 'Minimum Variance Angle'),
    'power-trace': ('Power Spectra Trace', WaveCalc.calc_power_trace, None,
        None, 'Power', 'nT^2/Hz', 'Trace Power Spectral Density'),
    'compressional-power': ('Compressional Power', WaveCalc.calc_compress_power,
        None, None, 'Power', 'nT^2/Hz', 'Compressional Power Spectral Density'),
}
POWER_TYPES = ('power-trace', 'compressional-power')

def extend_freqs(freqs, log_scale):
    ''' Adds the lower bounding frequency edge for the plot grid '''
    diff = abs(freqs[1] - freqs[0])
    lower = freqs[0] - diff
    if lower == 0 and log_scale:
        lower = freqs[0] - diff / 2
    return np.concatenate([[lower], freqs])

def run_window(session, start, end, vars, plot_type, fft_interval=None,
        fft_shift=None, bandwidth=3, detrend=False, scale='linear',
        vmin=None, vmax=None):
    series = session.get_many(vars, start, end)
    times = series[vars[0]].times
    data = np.vstack([series[v].data for v in vars])
    n = data.shape[1]
    res = session.resolution(vars[0])

    # Same defaults and checks as the GUI
    tool = DynamicAnalysisTool()
    interval = fft_interval if fft_interval is not None else tool.guess_fft_param(n)
    shift = fft_shift if fft_shift is not None else int(interval / 4)
    if shift < 1:
        raise CLIError(f'FFT shift must be at least 1 (got {shift}; '
            f'interval {interval} is too small for the default shift)')
    if interval <= shift:
        raise CLIError('FFT interval must be greater than the shift')
    if bandwidth % 2 == 0:
        raise CLIError('Bandwidth must be odd')
    if n <= interval:
        raise CLIError(f'Window has {n} points, which must exceed the FFT '
            f'interval ({interval})')

    gui_name, func, method, default_rng, label, units, title = PLOT_TYPES[plot_type]
    func_args = {} if method is None else {'method': method}
    fft_params = {'bandwidth': bandwidth, 'resolution': res, 'num_points': interval}

    time_stops, segs = tool.splitDataSegments(times, data, interval, shift)
    if detrend:
        segs = [np.vstack([signal.detrend(row) for row in seg]) for seg in segs]

    with np.errstate(invalid='ignore', divide='ignore'):
        columns = [func(seg, fft_params, **func_args) for seg in segs]
    grid = np.vstack(columns).T
    time_stops = np.array(time_stops)
    freqs = SpectraCalc.calc_freq(bandwidth, interval, res)

    log_color = plot_type in POWER_TYPES
    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else np.nanmin(grid)
        hi = vmax if vmax is not None else np.nanmax(grid)
        color_range = (lo, hi)
    elif default_rng is not None:
        color_range = default_rng
    else:
        positive = grid[grid > 0]
        color_range = (positive.min(), positive.max()) if positive.size else (1, 10)
    if color_range[0] >= color_range[1]:
        raise CLIError('Color range minimum must be less than the maximum')
    if log_color and color_range[0] <= 0:
        raise CLIError('Color range limits must be positive for power plots')

    log_freq = (scale == 'log')
    ticks_label = f'[{units}]' if units else None
    dates = ff_time.ticks_to_dates(time_stops, session.epoch)
    return {
        'plot_type': plot_type,
        'title': title,
        'samples': n,
        'resolution': res,
        'fft_interval': interval,
        'fft_shift': shift,
        'bandwidth': bandwidth,
        'detrend': bool(detrend),
        'frequency_scale': scale,
        'color_range': color_range,
        'log_color': log_color,
        'colorbar_label': [label] + ([ticks_label] if ticks_label else []),
        'times': [d.isoformat() for d in dates],
        'frequencies': freqs,
        'grid': grid,
        '_dates': dates,
        '_freq_edges': extend_freqs(freqs, log_freq),
    }

def render(session, values):
    return render_spectrogram(values)

def file_stem(params):
    return f"dynwave_{params['plot_type']}"

register(Tool(
    name='dynwave',
    description='Dynamic wave analysis spectrogram (image or data)',
    params=[
        Param('plot_type', 'choice', 'Which quantity do you want to plot?',
            default='ellipticity-means', choices=list(PLOT_TYPES), ask=True,
            labels={slug: info[0] for slug, info in PLOT_TYPES.items()}),
        Param('vars', 'vars', 'Three vector components, e.g. bx by bz or Bmso',
            required=True, count=3),
        Param('fft_interval', 'int', 'Points per FFT (default: chosen from the '
            'window length, as in the GUI)'),
        Param('fft_shift', 'int', 'Points to advance between FFTs (default: '
            'a quarter of the interval)'),
        Param('bandwidth', 'int', 'Frequency bands to average over; must be odd',
            default=3),
        Param('detrend', 'bool', 'Detrend each segment before the FFT'),
        Param('scale', 'choice', 'Frequency axis scale', default='linear',
            choices=['linear', 'log']),
        Param('vmin', 'float', 'Color scale minimum (values, not exponents, '
            'for power plots)'),
        Param('vmax', 'float', 'Color scale maximum'),
    ],
    run_window=run_window,
    formats=IMAGE_FORMATS + ('json',),
    default_format='png',
    render=render,
    file_stem=file_stem,
))
