import os

# Font parameters
default_font_size = 11 if os.name != 'nt' else 12
fonts = {
    'plot' : (None, default_font_size),
    'monospace' : ('monospace', 12) if os.name != 'nt' else 'Roboto Mono'
}