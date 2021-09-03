import itertools
import numpy as np

colors = {
    'red' : '#a60000',
    'green' : '#59bf00',
    'blue': '#0949b8',
    'purple': '#5509b8',
    'gold' : '#d19900',
    'black' : '#000000',
    'pink': '#d92b9c',
    'aqua': '#09b8b2',
}

def get_colors(plots):
    ''' Returns a list of colors associated with each trace in plot '''
    cvals = colors.values()
    maxn = max(list(map(len, plots)))
    cycle = itertools.cycle(cvals)
    plot_colors = []
    for row in plots:
        row_colors = []
        for dstr, en in row:
            if en >= 0:
                row_colors.append(next(cycle))
            else:
                row_colors.append(colors['black'])
        plot_colors.append(row_colors)
    return plot_colors

def get_segments(resolutions, avg_res):
    ''' Returns segments list for plotting traces with gaps'''
    mask = resolutions > (avg_res * 2)
    segments = np.array(np.logical_not(mask), dtype=np.int32)
    return segments