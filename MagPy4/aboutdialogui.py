# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'aboutdialog.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_aboutdialog(object):
    def setupUi(self, aboutdialog):
        aboutdialog.setObjectName("aboutdialog")
        aboutdialog.resize(380, 215)
        aboutdialog.setModal(True)
        self.buttonBox = QtWidgets.QDialogButtonBox(aboutdialog)
        self.buttonBox.setGeometry(QtCore.QRect(10, 173, 360, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.nameLabel = QtWidgets.QLabel(aboutdialog)
        self.nameLabel.setGeometry(QtCore.QRect(10, 10, 360, 40))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.versionLabel = QtWidgets.QLabel(aboutdialog)
        self.versionLabel.setGeometry(QtCore.QRect(10, 60, 360, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.versionLabel.setFont(font)
        self.versionLabel.setObjectName("versionLabel")
        self.copyrightLabel = QtWidgets.QLabel(aboutdialog)
        self.copyrightLabel.setGeometry(QtCore.QRect(10, 130, 360, 20))
        self.copyrightLabel.setObjectName("copyrightLabel")
        self.line = QtWidgets.QFrame(aboutdialog)
        self.line.setGeometry(QtCore.QRect(10, 160, 360, 3))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.retranslateUi(aboutdialog)
        self.buttonBox.accepted.connect(aboutdialog.accept)
        self.buttonBox.rejected.connect(aboutdialog.reject)
        QtCore.QMetaObject.connectSlotsByName(aboutdialog)

    def retranslateUi(self, aboutdialog):
        _translate = QtCore.QCoreApplication.translate
        aboutdialog.setWindowTitle(_translate("aboutdialog", "About"))
        self.nameLabel.setText(_translate("aboutdialog", "Name"))
        self.versionLabel.setText(_translate("aboutdialog", "Version"))
        self.copyrightLabel.setText(_translate("aboutdialog", "Copyright"))

