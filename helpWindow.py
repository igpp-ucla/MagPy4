
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np

class HelpWindowUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('MagPy4 Help')
        Frame.resize(700,500)  

        self.layout = QtWidgets.QVBoxLayout(Frame)
        self.text = QtGui.QTextBrowser()
        self.text.setText('''<HTML><head><style>* {white-space: pre; margin: 0; padding: 0;}</style></head><body>
<h2>Overview</h2>The MarsPy program is designed to simplify the analysis of magnetic field data.  It allows the user to examine the three components of the vector magnetic field and the total magnetic field as a function of time.  Once the interesting segment of data has been identified, the program enables the user to apply any one of a series of standard analysis tools to the data.  Characteristics of the data can be used to choose more natural coordinate systems in which to display the data.  The data may be filtered, Fourier analyzed etc. 
<h2>File</h2>The file module is used to select, open and add flatfiles you need.
<h2>Data</h2>The data module allows you to view all the data information contained in the flatfiles you choose. It also allows you to choose a specific time or a specific row and view the corresponding magnetic field data.
<h2>Plot</h2>The plot module allows you to add or remove plots. You can choose whatever components of the magnetic field data to plot. It also allows you to clear all plots. All plots are shown in the main plotting window.
<h2>Time Interval Selection Bar</h2>The time interval selection bar allows you to change the time interval displayed in the window. The top bar controls the start time of the time interval and the bottom bar controls the stop time of the time interval.
<h2>Trace Stats</h2>The trace stats menu will pop up when you click the start and stop time you are interested in on the main plotting window. It shows you the minimum value, maximum value, mean value, median value and standard deviation of the magnetic field data in the time interval you choose.
Suppose [x1, x2, ……xN] is the data set of one component of the vector magnetic field or the total magnetic field in the time interval you choose. Minimum is the smallest value of this data set and maximum is the largest value of this data set. The median is the value separating the higher half from the lower half of the data set. Suppose the mean value is 𝜇 and the standard deviation is σ,
<h2>Spectra</h2>The spectrum module gives you the power spectrum of the magnetic field signals, the coherence and the phase difference between any two components of the magnetic field in the time interval you choose. Moreover, the module allows you to do wave analysis.<h4>Power spectrum</h4>A digital signal containing n points sampled every ∆t is n∆t seconds long. The lowest frequency that can be accurately determined in this finite length time series is (n∆t)-1 and the highest is (2∆t)-1 because two data points are required to define a wave. This maximum frequency is called the Nyquist frequency. Fast Fourier transform routines can transform a time series of fluctuations of one component of magnetic field signals into a series of amplitudes of cosine and sine amplitudes, 
C0, C1,…… Cn/2-1         S1,, S2,……, Sn/2
If there are three orthogonal measurements of wave amplitudes in direction x, y, z, we can form the cospectral matrix of variances and cross variances at each frequency i:
This matrix is the cospectral density of the signal.
If averaging the sum of the diagonal terms over frequency, the plot would show the power spectral density of the signal versus frequency which is called the power frequency.<h4>Wave Analysis</h4>The cospectrum matrix is the real part of the spectrum matrix. The imaginary part of the spectrum matrix, which is called quaspectrum matrix, can be calculated at each frequency i by
These two matrixes give much information about wave properties. These properties were exploited by Born and Wolf (1970) to analyze optical signals and by Rankin and Kurtz (1970) to analyze micropulsations. Means (1972) showed that the direction of propagation can be directly obtained from the matrix Qi. <h4>Coherence</h4>The coherence between signals 1 and signals 2 is defined by. Where Cij is the ij-component of the cospectrum and Qij is the ij-component of the quaspectrum. <h4>Phase</h4>Phase is the difference, expressed in degrees or radians, between two waves having the same frequency and referenced to the same point in time. The 1,2 phase is defined as 
<h2>Edit</h2>The edit module allows you to do matrix rotation. You can either input custom rotation or click the minimum variance button to calculate the minimum variance matrix, and then transform the magnetic field vector to the new coordinate.<h4>Minimum variance</h4>By solving an eigenvalue problem, it is possible to calculate the rotation matrix that rotates a vector time series into a coordinate system in which the direction of the coordinate axes are those of the maximum variance, the minimum variance and an intermediate variance. The eigenvectors are the rows of the transformation matrix and the eigenvalues are the variances along each direction. These directions are called the principal axes. The system is also called minimum variance coordinate system. It can often be used at the magnetopause, but not at shocks. Usually the direction in which the magnetic field has a minimum variance is the direction normal to the magnetopause. Assume the data set you choose is [B1,B2,……BN] and construct a variance matrix M. Nine terms of M is mij (i and j represent the x, y, z components). Assume the eigenvalues of M is λ𝜇 (𝜇=1,2,3) and the eigenvectors that form the transformation matrix is V𝜇, MV𝜇 = λ𝜇 V𝜇
<h2>Options</h2>The option module allows you to change the style of the plot you want. You can choose to plot smooth lines of antialiasing lines. Or you can even only draw points. It also allows you to rescale y range to current time interval selection.
</body></HTML>''')

        self.layout.addWidget(self.text)
        
class HelpWindow(QtWidgets.QFrame, HelpWindowUI):
    def __init__(self, window, parent=None):
        super(HelpWindow, self).__init__(parent)

        self.window = window
        self.ui = HelpWindowUI()
        self.ui.setupUI(self, window)
