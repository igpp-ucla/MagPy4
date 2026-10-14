''' Registry of analysis tools that can be run from the command line.

    A tool declares its parameters (used to build both the command line flags
    and the interactive prompts) and a run_window function that computes its
    results for one time window. Results are plain Python values/numpy arrays
    so the output writers can format them without knowing about the tool.
'''
from dataclasses import dataclass, field

PARAM_KINDS = ('str', 'int', 'float', 'bool', 'choice', 'vars')

@dataclass
class Param():
    name: str
    kind: str = 'str'
    help: str = ''
    default: object = None
    required: bool = False
    choices: list = None # For kind == 'choice'
    count: int = None # Number of variables required for kind == 'vars'
    labels: dict = None # choice -> description shown in the menu
    ask: bool = False # Ask interactively even though there is a default

    def __post_init__(self):
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"Unknown parameter kind '{self.kind}'")

    @property
    def flag(self):
        return '--' + self.name.replace('_', '-')

DATA_FORMATS = ('json', 'csv', 'table')
IMAGE_FORMATS = ('png', 'pdf', 'svg')

@dataclass
class Tool():
    name: str
    description: str
    params: list = field(default_factory=list)
    run_window: object = None # run_window(session, start, end, **params) -> dict
    # Output formats the tool supports and the one used when none is given.
    # Tools that draw figures list image formats and set render, which turns
    # the values from run_window into a matplotlib Figure.
    formats: tuple = DATA_FORMATS
    default_format: str = 'json'
    render: object = None # render(session, values) -> Figure
    file_stem: object = None # file_stem(params) -> base name for image files

    def variables(self, values):
        ''' All variable names selected in the given parameter values '''
        names = []
        for p in self.params:
            if p.kind == 'vars' and values.get(p.name):
                names.extend(values[p.name])
        return names

TOOLS = {}

def register(tool):
    if tool.name in TOOLS:
        raise ValueError(f"Tool '{tool.name}' is already registered")
    TOOLS[tool.name] = tool
    return tool

def get_tool(name):
    return TOOLS.get(name)

def tool_names():
    return list(TOOLS.keys())

# Importing the modules registers the built-in tools
from . import stats
from . import spectra_wave
from . import dynwave
