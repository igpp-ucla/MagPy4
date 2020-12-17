from fflib import ff_reader
import re
from .general_data_import import FileData

class FF_FD():
    def __init__(self, filename, table):
        self.table = table
        self.name = filename

    def getFileType(self):
        return 'FLAT FILE'

    def getName(self):
        return self.name

    def getEpoch(self):
        return self.table.get_epoch()

    def getUnits(self):
        return self.table.get_units()

    def getLabels(self):
        return self.table.get_labels()

    def getRecords(self, epoch_var=None):
        return self.table

    def open(self):
        pass

    def close(self):
        pass

def load_flat_file(ff_path):
    # Get flat file reader object
    expr = '.+.ff[hd]*'
    if re.fullmatch(expr, ff_path):
        ff_path = '.'.join(ff_path.split('.')[:-1])
    ff = ff_reader(ff_path)

    data = ff.get_data_table()
    headers = ff.get_labels()
    epoch = ff.get_epoch()
    err_flag = ff.get_error_flag()
    units = ff.get_units()
    file_data = FileData(data, headers, epoch=epoch, error_flag=err_flag,
        units=units)
    reader = FF_FD(ff_path, file_data)

    return (file_data, reader)
