''' Basic per-variable statistics; a small reference tool that exercises the
    registry, window handling and output formats.
'''
import numpy as np
from . import Tool, Param, register

def run_window(session, start, end, vars):
    series = session.get_many(vars, start, end)
    values = {}
    for name, ts in series.items():
        values[name] = {
            'samples': len(ts.data),
            'mean': float(np.mean(ts.data)),
            'std': float(np.std(ts.data)),
            'min': float(np.min(ts.data)),
            'max': float(np.max(ts.data)),
        }
    return values

register(Tool(
    name='stats',
    description='Mean, standard deviation, min and max of variables',
    params=[Param('vars', 'vars', 'Variables to summarize', required=True)],
    run_window=run_window,
))
