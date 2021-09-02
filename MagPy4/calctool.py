from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
import re
from scipy.interpolate import CubicSpline
from bisect import bisect_left, bisect_right
import sys
from .layouttools import SplitterWidget
from functools import partial

class ListWidget(QtWidgets.QListWidget):
    ''' Editable list widget with delete triggers enabled '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shortcut = QtWidgets.QShortcut('delete', self)
        shortcut.activated.connect(self.deleteSelected)
        # self.setEditTriggers(QtWidgets.QListWidget.DoubleClicked)
        # self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.setAlternatingRowColors(True)
    
    def addItem(self, *args, **kwargs):
        ''' Modified original function to set all items editable by default '''
        super().addItem(*args, **kwargs)
        # for item in self.getItems():
            # item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
    
    def getItems(self):
        ''' Returns all items in widget '''
        count = self.count()
        items = [self.item(row) for row in range(count)]
        return items
    
    def getItemValues(self):
        ''' Get list of text items in list '''
        return [item.text() for item in self.getItems()]
    
    def deleteSelected(self):
        ''' Deletes selected rows '''
        items = self.selectedItems()
        indices = [self.indexFromItem(item) for item in items]
        indices = [index.row() for index in indices]
        indices = sorted(indices)[::-1]
        for index in indices:
            self.takeItem(index)

    def setItemData(self, index, data):
        item = self.item(index)
        item.setData(QtCore.Qt.UserRole, data)
        
    def getItemData(self, index):
        item = self.item(index)
        return item.data(QtCore.Qt.UserRole)

    def getItemDatas(self):
        return [self.getItemData(index) for index in range(self.count())]

class InputWindow(QtWidgets.QTextEdit):
    stmtSubmitted = QtCore.pyqtSignal(str)
    def __init__(self, output_func=None, prefix='>>> '):
        self.output_func = output_func
        self.prefix = prefix

        super().__init__()
    
    def setOutputFunc(self, func):
        self.output_func = func
    
    def keyPressEvent(self, ev):
        # If enter is pressed, submit the result
        if ev.key() == QtCore.Qt.Key_Return or ev.key() == QtCore.Qt.Key_Enter:
            super().keyPressEvent(ev)
            self.enterPressed()

        # If a non-movement key is pressed
        elif ev.key() < 0x01000000:
            # Add in prefix if initially on placeholder text
            if self.textCursor().position() <= 0 and self.toPlainText() == '':
                self.textCursor().insertText(self.prefix)
            
            # Move cursor to end of text
            self.moveCursor(QtGui.QTextCursor.End)
            super().keyPressEvent(ev)
        # Otherwise, use default key press event
        else:
            super().keyPressEvent(ev)

    def enterPressed(self):
        # Move text to last block start if not there
        self.moveCursor(QtGui.QTextCursor.End)
        self.moveCursor(QtGui.QTextCursor.StartOfBlock)
        self.moveCursor(QtGui.QTextCursor.PreviousBlock)

        # Get current text to submit and get output
        curr_text = self.textCursor().block().text()[len(self.prefix):]
        output = self.output_func(curr_text)

        # Move cursor to end
        self.moveCursor(QtGui.QTextCursor.End)

        # If there is output to submit, insert new block with it
        if output != '':
            self.textCursor().insertText(f'{output}')

        # Add in a new line with the given prefix
        if output != '':
            self.textCursor().insertBlock()
        self.textCursor().insertText(self.prefix)

        self.moveCursor(QtGui.QTextCursor.End)

class Vector():
    def __init__(self, values, times):
        self.times = times
        self.values = values
    
    def value(self):
        return (self.times, self.values)
    
    def interp(self, ref_times):
        ''' Interpolate vector along ref_times using Cubic Splines '''
        if np.array_equal(ref_times, self.times):
            return self.values
        else:
            cs = CubicSpline(self.times, self.values)
            return cs(ref_times)

    def __add__(self, o):
        if isinstance(o, Vector):
            vals = self.values + o.interp(self.times)
        else:
            vals = self.values + o
        
        return Vector(vals, self.times)

    def __sub__(self, o):
        if isinstance(o, Vector):
            vals = self.values - o.interp(self.times)
        else:
            vals = self.values - o
        
        return Vector(vals, self.times)

    def __mul__(self, o):
        if isinstance(o, Vector):
            vals = self.values * o.interp(self.times)
        else:
            vals = self.values * o
        
        return Vector(vals, self.times)

    def __truediv__(self, o):
        if isinstance(o, Vector):
            vals = self.values / o.interp(self.times)
        else:
            vals = self.values / o
        
        return Vector(vals, self.times)

    def __neg__(self):
        vals = self.values * (-1)
        return Vector(vals, self.times)
    
    def __pos__(self):
        return self
    
    def __pow__(self, p):
        vals = self.values ** p
        return Vector(vals, self.times)
    
    def __getitem__(self, i):
        return self.values[i]
    
    def __str__(self):
        return str(self.values)

    def func_map(f, v):
        if isinstance(v, Vector):
            return Vector(f(v.values), v.times)
        else:
            return f(v)

class simpleCalcUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Calculate')
        Frame.resize(400, 350)
        layout = QtWidgets.QGridLayout(Frame)

        # Input box and help window
        instrLbl = QtWidgets.QLabel('Please enter an expression to calculate:')
        self.helpBtn = QtWidgets.QPushButton('?')
        self.helpBtn.setMaximumWidth(20)

        instrFrm = QtWidgets.QFrame()
        instrLt = QtWidgets.QHBoxLayout(instrFrm)
        instrLt.setContentsMargins(0, 0, 0, 0)
        instrLt.addWidget(instrLbl)
        instrLt.addWidget(self.helpBtn)

        self.inputBox = InputWindow()
        info = '>>> BT = sqrt(BX_RTN^2 + BY_RTN^2 + BY_RTN^2)'
        info += '\n>>> avg = (BX + BY)/2'
        self.inputBox.setPlaceholderText(info)

        # Variable list
        outLbl = QtWidgets.QLabel('Variables')
        self.outputList = ListWidget()

        self.saveBtn = QtWidgets.QPushButton('Add Variables')
        self.saveBtn.setMaximumWidth(150)

        # Add to input/variable items to separate frames
        frames = []
        groups = [[instrFrm, self.inputBox], [outLbl, self.outputList, self.saveBtn]]
        for grp in groups:
            subframe = QtWidgets.QFrame()
            sublayout = QtWidgets.QVBoxLayout(subframe)
            sublayout.setContentsMargins(0, 0, 0, 0)
            for item in grp:
                sublayout.addWidget(item)
            frames.append(subframe)
        
        # Set alignment
        frames[1].layout().setAlignment(QtCore.Qt.AlignCenter)

        # Create splitter and add widgets to it
        splitter = SplitterWidget()
        splitter.setHandleWidth(20)
        splitter.setOrientation(QtCore.Qt.Vertical)

        splitter.addWidget(frames[0])
        splitter.addWidget(frames[1])
        layout.addWidget(splitter, 0, 0, 1, 1)

class simpleCalc(QtWidgets.QFrame, simpleCalcUI):
    def __init__(self, editWindow, window, parent=None):
        super(simpleCalc, self).__init__(parent)
        self.ui = simpleCalcUI()
        self.window = window
        self.editWindow = editWindow

        self.helpwin = None

        self.ui.setupUI(self, window)
        self.ui.inputBox.setOutputFunc(self.eval_wrapper)
        self.ui.saveBtn.clicked.connect(self.create_variables)
        self.ui.helpBtn.clicked.connect(self.showHelp)
        self.computed_vars = {}

    def showHelp(self):
        ''' Show help window '''
        self.closeHelp()

        # Create help window
        self.helpwin = QtWidgets.QWidget()
        self.helpwin.setWindowTitle('Calculate Tool Help')
        self.helpwin.resize(400, 500)
        layout = QtWidgets.QVBoxLayout(self.helpwin)
        widget = QtWidgets.QTextBrowser()
        widget.setReadOnly(True)
        layout.addWidget(widget)

        # Generate text
        txt = ("To create a variable type in a line of the form variable_name = "
            "math_expression and press enter.\n\n"
            "For example, to create a variable named Bt from the value produced by "
            "computing sqrt(Bx^2 + By^2 + Bz^2), type in\n"
            "Bt = sqrt(Bx^2 + By^2 + Bz^2)\n\n"
            "The list of variables that may be added will be shown in the "
            "box below. To delete a variable, click on it and press the 'delete' key."
            "To make the variables available for plotting. Click on the 'Add Variables' "
            "button.\n\n"
            "The following functions are available for use:\n")

        # Get math functions
        math_funcs = self.getMathDict()
        del math_funcs['pi']
        txt += '\n'.join(sorted(list(math_funcs.keys())))

        # Show text
        widget.setText(txt)
        self.helpwin.show()
    
    def closeHelp(self):
        if self.helpwin:
            self.helpwin.close()
            self.helpwin = None
    
    def closeEvent(self, ev):
        self.closeHelp()
        super().closeEvent(ev)

    def getMathDict(self):
        ''' Return a dictionary of common constants and functions in math '''
        varDict = {}

        # Add in pi constant
        varDict['pi'] = np.pi

        # Add in some basic trig functions
        for trigFunc in ['sin', 'cos', 'tan', 'arccos', 'arcsin', 'arctan',
            'rad2deg', 'deg2rad']:
            trig_func = getattr(np, trigFunc)
            varDict[trigFunc] = partial(Vector.func_map, trig_func)

        # Add in square root and natural log functions
        varDict['sqrt'] = partial(Vector.func_map, np.sqrt)
        varDict['log'] = partial(Vector.func_map, np.log)
    
        return varDict

    def getVectors(self, time_range=None):
        ''' Get a dictionary of Vector objects corresponding
            to 1D variables in main window 
        '''
        # Create a vector for each 1D variable in main window
        # and store in a dictionary
        en = self.window.currentEdit
        vecDict = {}
        for dstr in self.window.DATASTRINGS:
            data = self.window.getData(dstr, en)
            times = self.window.getTimes(dstr, en)[0]
            if time_range:
                tO, tE = time_range
                start = bisect_left(times, tO)
                end = bisect_right(times, tE)
                times = times[start:end]
                data = data[start:end]
            vec = Vector(data, times)
            vecDict[dstr] = vec
        
        return vecDict

    def evaluate(self, expr, time_range=None):
        ''' Evaluates an expression over the given time range,
            (full time range by default)
        '''
        # Replace ^ with ** equivalent
        expr = expr.replace('^', '**')

        # Get dictionary of vectors and math functions
        vecDict = self.getVectors(time_range)
        mathDict = self.getMathDict()
        vecDict.update(mathDict)
        vecDict.update(self.computed_vars)

        # Split variable name and expression
        if '=' not in expr:
            # Evaluate expression by itself
            try:
                value = eval(expr, vecDict)
                return (None, value)
            except:
                error = sys.exc_info()[0]
                return (None, f'Error: {error}')

        name, expr = expr.split('=')
        name = name.strip(' ')

        # Evalute expression
        try:
            value = eval(expr, vecDict)
        except:
            error = sys.exc_info()[0]
            return (None, f'Error: {error}')
    
        return (name, value)
    
    def eval_wrapper(self, expr):
        ''' Wrapper function for evaluate that returns result as a
            string and if applicable, saves the plot variable to the
            list of output variables
        '''
        # Don't do anything if the expression is empty
        if expr.strip(' ').strip('\n') == '':
            return ''

        # Evaluate the expression
        result = self.evaluate(expr)

        # Return the result as a string if no variable name is specified
        # or if an error resulted
        if result[0] is None:
            return str(result[1])
        else:
            # Skip invalid results
            if not isinstance(result[1], (int, float, Vector)):
                return ''

            # Add to output list if not yet added
            name, value = result
            if name not in self.computed_vars:
                self.ui.outputList.addItem(name)
                n = self.ui.outputList.count() - 1
                self.ui.outputList.setItemData(n, expr)

            # Save to dict of computed variables
            self.computed_vars[name] = value

            # Return an empty string
            return ''
        
    def create_variables(self):
        ''' Creates variables in the main window from the list
            of computed variables to add
        '''
        # Get list of variables to add and their corresponding
        # expressions
        varlst = self.ui.outputList.getItemValues()
        exprlst = self.ui.outputList.getItemDatas()

        # Try to add each variable to the main window
        for var, expr in zip(varlst, exprlst):
            if var not in self.computed_vars:
                continue
        
            # Assemble times and data values
            value = self.computed_vars[var]

            # Constants should be converted to a two value array
            if isinstance(value, (float, int)):
                tO, tE = self.window.tO, self.window.tE
                times = np.array([tO, tE])
                data = np.array([value, value])
            # Read in times and data directly from Vector objects
            else:
                times, data = value.value()
            
            # Add new variable to main window
            self.add_new_var(times, data, var, expr)
    
            # Clear output list
            self.ui.outputList.clear()

    def add_new_var(self, times, data, label, expr):
        ''' Adds variable data to main window entry and creates a new
            edit history entry
        '''
        if label in self.window.DATASTRINGS:
            self.window.DATADICT[label].append(data)
        else:
            diffs = np.diff(times)
            time_info = (times, diffs, np.mean(diffs))
            self.window.initNewVar(label, data, times=time_info)

        self.editWindow.addHistory(np.identity(3), expr, f'Calc {label}')
