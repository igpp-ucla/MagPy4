from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import numpy as np

class ExprElement():
    def isNum(self):
        return False

    def isVec(self):
        return False

    def isExpr(self):
        return False

class Num(ExprElement):
    def __init__(self, num):
        self.num = float(num)
        self.exprStr = num

    def isNum(self):
        return True

    def evaluate(self):
        return self.num

class Vec(ExprElement):
    def __init__(self, vec):
        self.vec = vec
        self.exprStr = str(vec)

    def isVec(self):
        return True

    def evaluate(self):
        return self.vec

class Var(ExprElement):
    def __init__(self, window, varName):
        self.varName = varName
        self.window = window
        self.exprStr = varName

    def isVec(self):
        return True

    def evaluate(self):
        if self.varName in self.window.DATASTRINGS:
            return np.array(self.window.getData(self.varName))
        else:
            raise Exception('Variable not in datastrings!')

class Operand(ExprElement):
    def __init__(self, opStr):
        self.opStr = opStr
        self.exprStr = opStr

    def calculate(self, lft, rght):
        # Evaluate left/right to num/array form and apply current operand
        result = None
        lftVal = lft.evaluate()
        rghtVal = rght.evaluate()

        if self.opStr == '+':
            result = lftVal + rghtVal
        elif self.opStr == '-':
            result = lftVal - rghtVal
        elif self.opStr == '*':
            result = lftVal * rghtVal
        elif self.opStr == '/':
            result = lftVal / rghtVal
        elif self.opStr == '^':
            result = lftVal ** rghtVal

        # Identify what kind of form the result should be returned in
        if lft.isVec() or rght.isVec():
            return Vec(result)
        return Num(result)

    def evaluate(self):
        return self.exprStr

class Expr(ExprElement):
    def __init__(self, exprList, window):
        self.exprList = exprList
        self.window = window
        self.exprStr = ''
        for e in exprList:
            self.exprStr += e

    def isExpr(self):
        return True

    def matchParens(self, exprLst, index):
        # Find the outermost matching ')' and create a subExpr from the expression
        # inside the parentheses; Also return index just after end of subExpr
        i = 0
        listLen = len(exprLst)
        for i in range(listLen-1, index, -1):
            if exprLst[i] == ')':
                subExpr = Expr(exprLst[index+1:i], self.window)
                return subExpr, i
        return None, None

    def evaluateStack(self, stack):
        # Evaluates operands in stack in PEMDAS order
        for opType in ['^', '*', '/', '+', '-']:
            index = 0
            while index < len(stack):
                e = stack[index]
                if e.exprStr == opType:
                    left = stack[index-1]
                    right = stack[index+1]
                    # Reduce expressions to vec/num before applying op to it
                    if left.isExpr():
                        left = left.evaluate()
                    if right.isExpr():
                        right = rght.evaluate()
                    # Update stack to insert newly calculated result in current position
                    stack = stack[:index-1] + [e.calculate(left, right)] + stack[index+2:]
                    index = index - 1
                index += 1

        # Eval any unevaluated expressions (in case stack contains a single Expr)
        index = 0
        for e in stack:
            if e.isExpr():
                stack[index] = e.evaluate()
            index += 1

        return stack

    def evaluate(self):
        # Organize expressions by parantheses and add ops/vars/nums to stack
        stack = []
        exprList = self.exprList
        index = 0
        while index < len(exprList):
            e = exprList[index]
            # If find '(', match w/ outer paranthesis and create a sub-Expr
            if e == '(':
                subExpr, i = self.matchParens(exprList, index)
                stack.append(subExpr)
                index = i
                continue
            # Convert numbers, variable names, and ops to corresp. ExprElem objects
            elif e.isdecimal():
                stack.append(Num(e))
            elif e.isalnum() or '_' in e:
                stack.append(Var(self.window, e))
            elif e in ['+', '-', '*', '/', '^']:
                stack.append(Operand(e))
            index += 1
        # Then evaluate items in stack accordingly by operand priority
        resultingStack = self.evaluateStack(stack)
        # Should only have one item (Vec/Num) to return in evaluated stack
        if len(resultingStack) == 1:
            return resultingStack[0]
        else:
            return None

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

    def splitString(self, expStr):
        # Split expression into list of strings for each op/parenthesis and variable name or number
        splitStr = []
        index = 0
        while index < len(expStr):
            c = expStr[index]
            # If character is an op, append to list
            if c in ['(', ')', '*', '+', '-', '/', '^', '=']:
                splitStr.append(c)
                index += 1
            # If character is a letter or number
            elif c.isalnum():
                numVarStr = ''
                # Loop through rest of string to get this variable/num's full string
                while index < len(expStr) and (expStr[index].isalnum() or (expStr[index] == '_')):
                    numVarStr += expStr[index]
                    index += 1
                splitStr.append(numVarStr)
            else:
                index += 1
        # Also identify variable to left of '=' where result will be stored
        varIndex = None
        index = 0
        for e in splitStr:
            if e == '=':
                varIndex = index - 1
            index += 1
        if varIndex != 0:
            return '', splitStr
        # Return storage variable, ordered list of ops/nums/vars in main expression
        return splitStr[0], splitStr[2:]

    def applyExpression(self):
        exprStr = self.ui.textBox.toPlainText()
        if exprStr == '':
            return
        # Break down expression into list of var/num/op strings and create an Expr obj
        self.varName, exprLst = self.splitString(exprStr)
        exprObj = Expr(exprLst, self.window)

        # Try evaluating exprObj, catch exceptions by printing error message to user
        try:
            result = exprObj.evaluate()
            if result.isNum():
                raise Exception('Setting array to a number')
            dta = result.evaluate()
            self.ui.statusBar.showMessage('Successfully evaluated...', 1000)
            # If successfully evaluted, add to edit history
            self.applyEdit(dta, exprStr)
        except:
            self.ui.statusBar.showMessage('Invalid expression!', 2000)

    def applyEdit(self, dta, exprStr):
        # Add to current var's edited data
        if self.varName in self.window.DATASTRINGS:
            self.window.DATADICT[self.varName].append(dta)
            self.editWindow.addHistory(np.identity(3), exprStr, 'Calc')
        elif self.varName.lower() in [dstr.lower() for dstr in self.window.DATASTRINGS]:
            # If lowercase varname == lowercase existing datastring, raise an error msg
            self.ui.statusBar.showMessage('Error: Invalid variable name.')
        else:
            # If not in datastrings, create a new variable
            self.initNewVar(self.varName, dta)
            self.editWindow.addHistory(np.identity(3), exprStr, 'Calc ' + self.varName)
            self.ui.statusBar.showMessage('New variable '+self.varName+' created.', 3000)

    def initNewVar(self, dstr, dta):
        # Add new variable name to list of datastrings
        self.window.DATASTRINGS.append(dstr)
        self.window.ABBRV_DSTR_DICT[dstr] = dstr

        # Use any datastring's times as base
        times = self.window.getTimes(self.window.DATASTRINGS[0], 0)
        self.window.TIMES.append(times)
        self.window.TIMEINDEX[dstr] = len(self.window.TIMES) - 1

        # Add in data to dictionaries, no units
        self.window.ORIGDATADICT[dstr] = dta
        self.window.DATADICT[dstr] = [dta]
        self.window.UNITDICT[dstr] = ''

        # Pad rest of datadict to have same length
        length = len(self.editWindow.history)
        while len(self.window.DATADICT[dstr]) < length:
            self.window.DATADICT[dstr].append([])