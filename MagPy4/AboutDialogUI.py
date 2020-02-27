# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AboutDialog.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AboutDialog(object):
    def setupUi(self, AboutDialog):
        AboutDialog.setObjectName("AboutDialog")
        AboutDialog.resize(380, 215)
        AboutDialog.setModal(True)
        self.buttonBox = QtWidgets.QDialogButtonBox(AboutDialog)
        self.buttonBox.setGeometry(QtCore.QRect(10, 173, 360, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.nameLabel = QtWidgets.QLabel(AboutDialog)
        self.nameLabel.setGeometry(QtCore.QRect(10, 10, 360, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.versionLabel = QtWidgets.QLabel(AboutDialog)
        self.versionLabel.setGeometry(QtCore.QRect(10, 60, 360, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.versionLabel.setFont(font)
        self.versionLabel.setObjectName("versionLabel")
        self.copyrightLabel = QtWidgets.QLabel(AboutDialog)
        self.copyrightLabel.setGeometry(QtCore.QRect(10, 130, 360, 20))
        self.copyrightLabel.setObjectName("copyrightLabel")
        self.line = QtWidgets.QFrame(AboutDialog)
        self.line.setGeometry(QtCore.QRect(10, 160, 360, 3))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.retranslateUi(AboutDialog)
        self.buttonBox.accepted.connect(AboutDialog.accept)
        self.buttonBox.rejected.connect(AboutDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(AboutDialog)

    def retranslateUi(self, AboutDialog):
        _translate = QtCore.QCoreApplication.translate
        AboutDialog.setWindowTitle(_translate("AboutDialog", "About"))
        self.nameLabel.setText(_translate("AboutDialog", "Name"))
        self.versionLabel.setText(_translate("AboutDialog", "Version"))
        self.copyrightLabel.setText(_translate("AboutDialog", "Copyright"))

