import numpy as np
from scipy import constants
logbase_map = {'10' : np.log10, '2': np.log2, 'e' : np.log }
logbase_reverse = {
    '10' : lambda x : 10 ** x,
    '2' : lambda x : 2 ** x,
    'e' : lambda x : np.exp(x)
}