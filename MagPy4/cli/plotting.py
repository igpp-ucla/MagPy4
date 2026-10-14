''' matplotlib rendering of figures for the command line.

    Figures are built with matplotlib.figure.Figure directly (no pyplot or GUI
    backend), so they can be saved from any environment without a display.
'''
import numpy as np
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure

# The GUI's RGB gradient (RGBGradient in tools/dynbase.py)
GUI_GRADIENT = [
    (0, (25, 0, 245)),
    (1 / 3, (0, 245, 245)),
    (0.5, (50, 245, 0)),
    (2 / 3, (245, 245, 0)),
    (1, (245, 0, 25)),
]

def gui_colormap():
    stops = [(pos, tuple(c / 255 for c in rgb)) for pos, rgb in GUI_GRADIENT]
    return LinearSegmentedColormap.from_list('magpy_rgb', stops)

def render_spectrogram(values, figsize=(9, 6)):
    ''' Draws the spectrogram described by the values of the dynwave tool:
        'grid' (frequency rows by time columns), the time column edges in
        '_dates', frequency edges in '_freq_edges', and the plot settings
    '''
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    x = mdates.date2num(values['_dates'])
    y = np.asarray(values['_freq_edges'])
    z = np.ma.masked_invalid(np.asarray(values['grid'], dtype=float))
    vmin, vmax = values['color_range']
    log_color = values['log_color']
    if log_color:
        z = np.ma.masked_less_equal(z, 0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(x, y, z, cmap=gui_colormap(), norm=norm, shading='flat')
    bar = fig.colorbar(mesh, ax=ax, pad=0.03, aspect=17)
    bar.set_label('\n'.join(values['colorbar_label']))

    log_freq = values['frequency_scale'] == 'log'
    if log_freq:
        ax.set_yscale('log')
    ax.set_ylabel('Log Frequency (Hz)' if log_freq else 'Frequency (Hz)')
    ax.set_title(values['title'])

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(top=True, right=True, length=4)
    ax.margins(x=0)

    fmt = '%Y %b %d %H:%M:%S'
    start = values['_dates'][0].strftime(fmt)
    end = values['_dates'][-1].strftime(fmt)
    fig.text(0.08, 0.015, f'Time Range: {start} to {end}')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig
