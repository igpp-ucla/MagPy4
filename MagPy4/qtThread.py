from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import traceback, sys

class ThreadSignals(QtCore.QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class TaskRunner(QtCore.QRunnable):
    def __init__(self, fn, *args, update_progress=False, **kwargs):
        super(TaskRunner, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = ThreadSignals()

        if update_progress:
            self.kwargs['progress_func'] = self.update_progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args, **self.kwargs, 
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            self.signals.finished.emit()  # Done
        else:
            self.signals.result.emit(result)  # Return the result of the processing

    def update_progress(self, val):
        self.signals.progress.emit(val)

