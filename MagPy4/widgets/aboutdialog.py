# aboutdialog.py - About dialog box
#
# The About dialog box user interface was designed using Qt Designer, which
# writes out the file aboutdialog.ui. This file is in turn converted into
# aboutdialogui.py at the command-line using the command:
#
# pyuic5 aboutdialog.ui -o aboutdialogui.py

from .aboutdialogui import Ui_aboutdialog
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

class aboutdialog(QtWidgets.QDialog, Ui_aboutdialog):

    def __init__(self, name, version, copyright, parent=None):
        super(aboutdialog, self).__init__()

        self.ui = Ui_aboutdialog()
        self.ui.setupUi(self)

        self.setWindowTitle(f'About {name}')

        self.ui.nameLabel.setText(name)
        self.ui.versionLabel.setText(version)
        self.ui.copyrightLabel.setText(copyright)

        self.parent = parent

        # Remove the question mark (context help button) from the dialog box's title bar.
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        # Wire up the OK button.
        self.accepted.connect(self.onAccepted)

    def onAccepted(self):
        """Called when the user clicks the OK button
        """
        self.close()
