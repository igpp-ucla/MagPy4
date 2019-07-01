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

    def __eq__(self, val): 
        if self.exprStr == val: 
            return True
        return False

class Num(ExprElement):
    def __init__(self, num):
        self.num = float(num)
        self.exprStr = str(num)

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
    def __init__(self, window, varName, dataRange=None):
        self.varName = varName
        self.window = window
        self.exprStr = varName
        self.dataRange = dataRange # Limits the data range of operations

    def isVec(self):
        return True

    def evaluate(self):
        if self.varName in self.window.DATASTRINGS:
            if self.dataRange is None: # Operation on full set of data
                return np.array(self.window.getData(self.varName))
            else: # Operation on a range of data
                iO, iE = self.dataRange
                return np.array(self.window.getData(self.varName)[iO:iE])
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
    def __init__(self, stackLst):
        self.exprStack = stackLst
        self.exprStr = ''
        for e in stackLst:
            self.exprStr += e.exprStr

    def isExpr(self):
        return True

    def evaluate(self):
        if len(self.exprStack) == 1:
            return self.exprStack[0]
        elif len(self.exprStack) == 0 or len(self.exprStack) == 2:
            return None

        # Evaluate the items on the stack in PEMDAS order
        for opType in ['^', '*', '/', '+', '-']:
            index = 0
            resultsStack = []
            while index < len(self.exprStack):
                e = self.exprStack[index]
                # If op found, calculate the current sub-expr and push onto stack
                if e == opType:
                    left = resultsStack.pop()
                    right = self.exprStack[index+1]
                    if left.isExpr():
                        left = left.evaluate()
                    if right.isExpr():
                        right = right.evaluate()
                    res = e.calculate(left, right)
                    resultsStack.append(res)
                    # Prev element was popped off stack, skip next element too
                    index += 2
                # If not an op, just push on to stack
                else:
                    resultsStack.append(e)
                    index += 1
            # Update the main stack with the resulting stack just calculated
            self.exprStack = resultsStack

        return self.exprStack[0]

class ExpressionEvaluator():
    '''  Encapsulates tools used for creating and evaluating an expression
        from a user-provided string
    '''
    def __init__(self, exprStr, window, dataRange=None):
        self.exprStr = exprStr
        self.window = window
        self.dataRange = dataRange # Optional data range limits Var element data ranges

    def splitByOp(self, s):
        # Looks for every operand and splits the previously cleaned expression 
        # string by the operands into a list
        lst = []
        lastIndex = 0
        opList = ['+', '-', '*', '/', '^', ')', '(']
        for i in range(0, len(s)):
            if s[i] in opList:
                subStr = s[lastIndex:i]
                if subStr != '':
                    lst.append(subStr)
                lst.append(s[i])
                lastIndex = i + 1
        # Add in any elements after last operand
        if s[i] not in opList:
            lst.append(s[lastIndex:i+1])
        return lst

    def splitString(self):
        expStr = self.exprStr

        # Split expression into list of strings for each op/parenthesis and variable name or number
        cleanExpr = expStr.strip('\n').replace(' ', '')

        # Make sure there is a variable to be set
        if cleanExpr.count('=') != 1:
            return None, self.exprStr

        # Extract the var name from left hand side of eq sign and break
        # down the expression into operators, numbers, and variable names
        varName, expStr = cleanExpr.split('=')
        exprLst = self.splitByOp(expStr)

        # Return storage variable, ordered list of ops/nums/vars in main expression
        return varName, exprLst

    def isNumber(self, e):
        cleanElem = e.replace('.', '').replace('-','').replace('+','')
        counts = [cleanElem.count(c) for c in ['+','-','.']]

        # Checks if no more than one period and only one sign character
        validCounts = True
        if max(counts) >= 1 or counts[0] + counts[2] > 1:
            validCounts = False

        if cleanElem.isnumeric() and validCounts:
            return True
        else:
            return False

    def mapToObj(self, e):
        # Map string to an ExprElement object
        obj = None
        if e in ['+', '-', '*', '/', '^']:
            obj = Operand(e)
        elif self.isNumber(e):
            obj = Num(e)
        elif e.isalnum() or '_' in e:
            obj = Var(self.window, e, self.dataRange)
        return obj

    def createStack(self, exprLst):
        exprStack = []
        listLen = len(exprLst)

        if exprLst.count('(') != exprLst.count(')'):
            raise Exception('Unmatched parentheses!')

        # Creates a stack of ExprElem objects from split expression string
        for i in range(0, listLen):
            c = exprLst[i]
            if c == '(':
                exprStack.append(c)
            elif c == ')':
                # If closing parens found, pop eveything off stack until
                # an open parens is found and create a subExpr from popped items
                subStack = []
                for i in range(0, i):
                    currOp = exprStack.pop()
                    if currOp == '(':
                        break
                    subStack.append(currOp)
                subStack.reverse()

                subExpr = Expr(subStack)
                exprStack.append(subExpr)
            else:
                # Otherwise, just map the element to an ExprElem object
                obj = self.mapToObj(c)
                exprStack.append(obj)

        return Expr(exprStack)

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

    def applyExpression(self):
        exprStr = self.ui.textBox.toPlainText()
        if exprStr == '':
            return
        # Break down expression into list of var/num/op strings and create an Expr obj
        expEval = ExpressionEvaluator(exprStr, self.window)
        self.varName, exprLst = expEval.splitString()
        exprObj = expEval.createStack(exprLst)

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
            self.window.initNewVar(self.varName, dta)
            self.editWindow.addHistory(np.identity(3), exprStr, 'Calc ' + self.varName)
            self.ui.statusBar.showMessage('New variable '+self.varName+' created.', 3000)