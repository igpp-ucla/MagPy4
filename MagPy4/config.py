import os

# Font parameters
unix_machine = (os.name not in ['nt', 'posix'])
default_font_size = 11 if unix_machine else 12
default_mono_font = None if unix_machine else 'Roboto Mono'
fonts = {
    'plot' : (None, default_font_size),
    'monospace' : (default_mono_font, default_font_size)
}