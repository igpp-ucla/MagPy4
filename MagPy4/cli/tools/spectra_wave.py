''' Wave analysis of a three-component vector over a time window.

    Computes the same values as the Open Wave Analysis window of the Spectra
    tool (see WaveAnalysis.updateCalculations in tools/waveanalysis.py): the FFT of
    each component is taken over the window, the spectral matrix is summed over
    a band of frequency bins, and SpecWave derives the wave parameters from it.
'''
import numpy as np
from ...alg.spectraalg import SpectraCalc, SpecWave
from ..prompt import CLIError
from . import Tool, Param, register

# SpecWave also produces SVD-method values (svd_*), but the GUI never shows
# them and svd_prop_angle is computed from the un-normalized mean field, so
# they are left out of the results

def resolve_band(n, res, fmin=None, fmax=None, bin_min=None, bin_max=None):
    ''' Returns the FFT bin slice (lo, hi), hi exclusive, for the band.

        Bin i has frequency i / (n * res), so the band in Hz maps to bins
        by rounding inwards. bin_min/bin_max are raw bin indices, as in the
        GUI's Frequency Range boxes (which index the FFT array directly);
        they take precedence over fmin/fmax. By default the band runs from
        the first bin above 0 Hz to the Nyquist frequency; the GUI's default
        starts at bin 0, which puts the mean field into the spectral matrix.
    '''
    nbins = n // 2 + 1
    df = 1.0 / (n * res)

    if bin_min is not None:
        lo = bin_min
    elif fmin is not None:
        lo = int(np.ceil(fmin / df - 1e-9))
    else:
        lo = 1

    if bin_max is not None:
        hi = bin_max
    elif fmax is not None:
        hi = int(np.floor(fmax / df + 1e-9)) + 1
    else:
        hi = nbins

    hi = min(hi, nbins)
    if lo < 0:
        raise CLIError('Frequency band starts below 0 Hz')
    if hi - lo < 2:
        raise CLIError(
            f'Frequency band contains {max(hi - lo, 0)} FFT bin(s) but at least 2 '
            f'are needed. For this window the bin spacing is {df:.6g} Hz and '
            f'frequencies run up to {(nbins - 1) * df:.6g} Hz')
    return lo, hi

def matrix(mat):
    ''' Real and imaginary parts of a complex matrix, as the GUI shows them '''
    mat = np.asarray(mat)
    return {'real': mat.real, 'imag': mat.imag}

def run_window(session, start, end, vars, fmin=None, fmax=None,
        fmin_index=None, fmax_index=None):
    series = session.get_many(vars, start, end)
    data = [series[v].data for v in vars]
    n = len(data[0])
    res = session.resolution(vars[0])

    lo, hi = resolve_band(n, res, fmin, fmax, fmin_index, fmax_index)

    ffts = [SpectraCalc.calc_fft(d) for d in data]
    avg = np.mean(np.vstack(data), axis=1)
    params = {'num_points': n, 'resolution': res, 'freq_range': (lo, hi)}

    with np.errstate(invalid='ignore', divide='ignore'):
        p = SpecWave().get_params(ffts, params, avg)

    df = 1.0 / (n * res)
    return {
        'samples': n,
        'resolution': res,
        'frequency_band': {'bin_start': lo, 'bin_stop': hi, 'bins': hi - lo,
            'min_hz': lo * df, 'max_hz': (hi - 1) * df, 'spacing_hz': df},
        'mean_field': avg,
        'power': {'cospectral': p['sum_mat'].real, 'quadrature': p['sum_mat'].imag},
        'transformed_power': matrix(p['bw_transf_pow']),
        'transformed_matrix': matrix(p['means_transf_mat']),
        'eigenvalues': p['bw_eigenvalues'],
        'eigenvectors': p['bw_eigenvectors'],
        'propagation_vector': p['means_k'],
        'propagation_angle': p['means_prop_angle'],
        'linear_variance_vector': p['lin_var_vec'],
        'linear_variance_angle': p['lin_var_angle'],
        'ellipticity': {'born_wolf': p['bw_ellip'], 'means': p['means_ellip']},
        'percent_polarization': {'born_wolf': p['bw_perc_polar'],
            'means': p['means_perc_polar']},
        'azimuth_angle': p['means_azim_angle'],
    }

register(Tool(
    name='spectra-wave',
    description="Spectra tool's Wave analysis of one time range (polarization, ellipticity, propagation direction...)",
    params=[
        Param('vars', 'vars', 'Three vector components, e.g. bx by bz or Bmso',
            required=True, count=3),
        Param('fmin', 'float', 'Lower edge of the frequency band in Hz '
            '(default: first bin above 0 Hz)'),
        Param('fmax', 'float', 'Upper edge of the frequency band in Hz '
            '(default: Nyquist frequency)'),
        Param('fmin_index', 'int', 'First FFT bin of the band; overrides --fmin '
            '(same as the GUI Frequency Range minimum)'),
        Param('fmax_index', 'int', 'End FFT bin of the band, exclusive; overrides '
            '--fmax (same as the GUI Frequency Range maximum)'),
    ],
    run_window=run_window,
))
