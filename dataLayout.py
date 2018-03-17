
DATATABLE = {
    'BX1': 0, 'BY1': 1, 'BZ1': 2, 'BT1': 3, 'PX1': 4, 'PY1': 5, 'PZ1': 6, 'PT1': 7,
    'BX2': 8, 'BY2': 9, 'BZ2':10, 'BT2':11, 'PX2':12, 'PY2':13, 'PZ2':14, 'PT2':15,
    'BX3':16, 'BY3':17, 'BZ3':18, 'BT3':19, 'PX3':20, 'PY3':21, 'PZ3':22, 'PT3':23,
    'BX4':24, 'BY4':25, 'BZ4':26, 'BT4':27, 'PX4':28, 'PY4':29, 'PZ4':30, 'PT4':31,
    'JXM':32, 'JYM':33, 'JZM':34, 'JTM':35, 'JPARA':36, 'JPERP':37, 'JANGLE':38
    # i think every column is mapped
}
 # dict of lists, key is data string below, value is list of data to plot
DATADICT = {}
# list of each field wanted to plot
DATASTRINGS = ['BX1','BX2','BX3','BX4',
               'BY1','BY2','BY3','BY4',
               'BZ1','BZ2','BZ3','BZ4',
               'BT1','BT2','BT3','BT4',
               'JXM','JYM','JZM','JTM','JPARA','JPERP','JANGLE',
               'VEL']

CALCTABLE = {
    #'VEL':calcVel    
    #'CURL'
    #'PRESSURE'
    #'DENSITY'
}
