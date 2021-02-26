
default_barlabel = 'Power (nT^2/Hz)'
spec_infos = {
    'spec' : {
        'name' : 'Spectrogram',
        'legend_label' : default_barlabel,
        'color_rng' : None,
    },
    'coh' : {
        'name' : 'Coherence',
        'legend_label' : 'Coherence',
        'color_rng' : (0, 1.0),
    },
    'pha' : {
        'name' : 'Phase',
        'legend_label' : 'Angle (Degrees)',
        'color_rng' : (-180, 180),
    },
    'ellip' : {
        'name' : 'Ellipticity (SVD)',
        'legend_label' : 'Ellipticity',
        'color_rng' : (0, 1.0),
    },
    'propangle' : {
        'name' : 'Propagation Angle (SVD)',
        'legend_label' : 'Angle (Degrees)',
        'color_rng' : (0, 90),
    },
    'powertrace' : {
        'name' : 'Trace Power Spectral Density',
        'legend_label' : default_barlabel,
        'color_rng' : None,
    },
    'comppower' : {
        'name' : 'Compressional Power Spectral Density',
        'legend_label' : default_barlabel,
        'color_rng' : None,
    },
    'tranvpower' : {
        'name' : 'Tranverse Power',
        'legend_label' : default_barlabel,
        'color_rng' : None,
    }
}