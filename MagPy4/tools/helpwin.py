# helpwin.py - Help window
#
# Displays the program's online help.

from pathlib import Path

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtPrintSupport
from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
from .. import get_relative_path

class HelpWindowUI(object):
    def setupUI(self, frame, window):
        frame.setWindowTitle('MagPy4 Help')
        frame.resize(800, 600)

        rsrc_dir = get_relative_path('rsrc')
        htmlPath = os.path.join(rsrc_dir, 'help', 'help.html')
        html = Path(htmlPath).read_text()

        cssPath = os.path.join(rsrc_dir, 'help', 'help.css')
        cssPath = os.path.join()
        self.view = QWebEngineView()
        self.view.setHtml(html, QtCore.QUrl.fromLocalFile(cssPath))

        self.layout = QtWidgets.QVBoxLayout(frame)
        self.layout.addWidget(self.view)

        self.printButton = QtWidgets.QPushButton('&Print...')
        self.printButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.printButton.clicked.connect(self.printHelp)

        self.bottomLayout = QtWidgets.QHBoxLayout()
        self.bottomLayout.addStretch(1)
        self.bottomLayout.addWidget(self.printButton)

        self.layout.addLayout(self.bottomLayout)

    def printHelp(self):
        """ Prints the program's online help.
        """
        self.printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.ScreenResolution)
        self.printer.setPageMargins(1.0, 1.0, 1.0, 1.0, QtPrintSupport.QPrinter.Inch)
        self.printDialog = QtPrintSupport.QPrintDialog(self.printer)
        if self.printDialog.exec_() == QtPrintSupport.QPrintDialog.Accepted:
            self.view.page().print(self.printer, self.callback)

    def callback(self, success):
        if success == True:
            pass
            # Help was printed successfully.
        else:
            pass
            # Help wasn't printed.

class HelpWindow(QtWidgets.QFrame, HelpWindowUI):
    def __init__(self, window, parent=None):
        super(HelpWindow, self).__init__(parent)

        self.window = window
        self.ui = HelpWindowUI()
        self.ui.setupUI(self, window)
