from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
import re
from scipy.interpolate import CubicSpline
from bisect import bisect_left, bisect_right

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
    
class simpleCalcUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Calculate')
        Frame.resize(400, 150)
        layout = QtWidgets.QGridLayout(Frame)

        self.instrLbl = QtWidgets.QLabel('Please enter an expression to calculate:\n')

        self.textBox = QtWidgets.QTextEdit()
        exampleTxt = 'Examples:\nBx_IFG = Bx_IFG * 3 + 5^2\nBx_Avg = (BX_GSM1 + BX_GSM2)/2'
        self.textBox.setPlaceholderText(exampleTxt)

        self.applyBtn = QtWidgets.QPushButton('Apply')
        self.applyBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizeGripEnabled(False)

        layout.addWidget(self.instrLbl, 0, 0, 1, 2)
        layout.addWidget(self.textBox, 1, 0, 1, 2)
        layout.addWidget(self.applyBtn, 2, 1, 1, 1)
        layout.addWidget(self.statusBar, 2, 0, 1, 1)

class simpleCalc(QtGui.QFrame, simpleCalcUI):
    def __init__(self, editWindow, window, parent=None):
        super(simpleCalc, self).__init__(parent)
        self.ui = simpleCalcUI()
        self.window = window
        self.editWindow = editWindow

        self.ui.setupUI(self, window)
        self.ui.applyBtn.clicked.connect(self.applyExpression)

    def getMathDict(self):
        ''' Return a dictionary of common constants and functions in math '''
        varDict = {}

        # Add in pi constant
        varDict['pi'] = np.pi

        # Add in some basic trig functions
        for trigFunc in ['sin', 'cos', 'tan', 'arccos', 'arcsin', 'arctan']:
            varDict[trigFunc] = getattr(np, trigFunc)

        # Add in square root and natural log functions
        varDict['sqrt'] = np.sqrt
        varDict['log'] = np.log
    
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
        # Split variable name and expression
        if '=' not in expr:
            return None
        
        name, expr = expr.split('=')
        name = name.strip(' ')

        # Replace ^ with ** equivalent
        expr = expr.replace('^', '**')

        # Get dictionary of vectors and math functions
        vecDict = self.getVectors(time_range)
        mathDict = self.getMathDict()
        vecDict.update(mathDict)

        # Evalute expression
        try:
            value = eval(expr, vecDict)
        except:
            return None
        
        return (name, value)

    def applyExpression(self):
        exprStr = self.ui.textBox.toPlainText()
        if exprStr == '':
            return

        # Break down expression into list of var/num/op strings and create an Expr obj
        result = self.evaluate(exprStr)

        # Try evaluating exprObj, catch exceptions by printing error message to user
        if result is not None:
            self.ui.statusBar.showMessage('Successfully evaluated...', 1000)
            # If successfully evaluted, add to edit history
            self.applyEdit(result, exprStr)
        else:
            self.ui.statusBar.showMessage('Invalid expression!', 2000)

    def applyEdit(self, result, exprStr):
        # Add to current var's edited data
        varName, data = result
        if varName in self.window.DATASTRINGS:
            self.window.DATADICT[varName].append(data)
            self.editWindow.addHistory(np.identity(3), exprStr, 'Calc')
        elif varName.lower() in [dstr.lower() for dstr in self.window.DATASTRINGS]:
            # If lowercase varname == lowercase existing datastring, raise an error msg
            self.ui.statusBar.showMessage('Error: Invalid variable name.')
        else:
            # If not in datastrings, create a new variable
            times, values = data.value()
            diff = np.diff(times)
            timeInfo = (times, diff, np.median(diff))
            self.window.initNewVar(varName, values, times=timeInfo)
            self.editWindow.addHistory(np.identity(3), exprStr, 'Calc ' + varName)
            self.ui.statusBar.showMessage('New variable '+varName+' created.', 3000)