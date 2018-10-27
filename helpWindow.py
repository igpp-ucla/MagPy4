# helpWindow.py - Help window
#
# Displays the program's help file.

from pathlib import Path

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView

class HelpWindowUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('MagPy4 Help')
        Frame.resize(700,500)  

        self.layout = QtWidgets.QVBoxLayout(Frame)
        self.text = QWebEngineView()

        html = Path('help.html').read_text()

        self.text.setHtml(html, QtCore.QUrl.fromLocalFile('\\help.css'))
        self.layout.addWidget(self.text)
        
class HelpWindow(QtWidgets.QFrame, HelpWindowUI):
    def __init__(self, window, parent=None):
        super(HelpWindow, self).__init__(parent)

        self.window = window
        self.ui = HelpWindowUI()
        self.ui.setupUI(self, window)
