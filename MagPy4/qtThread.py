from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import traceback, sys
import functools

class ThreadSignals(QtCore.QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)

class ThreadPool():
    ''' Manages a pool of threads that run TaskThreads '''
    def __init__(self):
        self.threads = [] # Current threads in queue

    def clear(self):
        ''' Attempts to cancel all threads and clears them from list '''
        self.terminate()
        self.threads = []

    def terminate(self):
        ''' Requests all threads to be interrupted '''
        for thread in self.threads:
            thread.requestInterruption()

    def start(self, task):
        ''' Starts running a thread and connects its finish function
            to update_threads function
        '''
        self.threads.append(task)
        updt_func = functools.partial(self.update_threads, task)
        task.signals.finished.connect(updt_func)
        task.start()
    
    def update_threads(self, thread):
        ''' Removes given thread from list of threads in queue '''
        if thread in self.threads:
            self.threads.remove(thread)
            thread.deleteLater()
        
class TaskThread(QtCore.QThread):
    ''' Thread with customized run function similar to TaskRunner below '''
    def __init__(self, fn, *args, update_progress=False, interrupt_enabled=False, **kwargs):
        super().__init__()
        self.setTerminationEnabled(True)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = ThreadSignals()

        if update_progress:
            self.kwargs['progress_func'] = self.update_progress
        
        if interrupt_enabled:
            self.kwargs['cancel_func'] = self.isInterruptionRequested

    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs, 
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        self.signals.finished.emit()
    
    def update_progress(self, val):
        self.signals.progress.emit(val)
    
class TaskRunner(QtCore.QRunnable):
    def __init__(self, fn, *args, update_progress=False, **kwargs):
        super(TaskRunner, self).__init__()
        self.setAutoDelete(True)

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = ThreadSignals()

        if update_progress:
            self.kwargs['progress_func'] = self.update_progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs, 
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            self.signals.finished.emit()
        else:
            self.signals.result.emit(result)

    def update_progress(self, val):
        self.signals.progress.emit(val)