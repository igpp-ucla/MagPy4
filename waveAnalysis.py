
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
from mth import Mth
from MagPy4UI import MatrixWidget

class WaveAnalysisUI(object):
        def setupUI(self, Frame, window):
            Frame.setWindowTitle('Wave Analysis')
            Frame.resize(1000,700)  

            self.layout = QtWidgets.QVBoxLayout(Frame)

            self.axLayout = QtWidgets.QGridLayout()
            self.window = window
            defaultPlots = self.window.getDefaultPlotInfo()[0]
            axes = ['X','Y','Z','T']
            self.axesDropdowns = []
            for i,ax in enumerate(axes):
                dd = QtGui.QComboBox()
                self.axLayout.addWidget(QtWidgets.QLabel(ax),0,i,1,1)
                for s in self.window.DATASTRINGS:
                    if ax.lower() in s.lower():
                        dd.addItem(s)
                dd.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                self.axesDropdowns.append(dd)
                self.axLayout.addWidget(dd,1,i,1,1)

            spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            self.axLayout.addItem(spacer, 0, 100, 1, 1)

            self.layout.addLayout(self.axLayout)
            
            rpFrame = QtWidgets.QGroupBox('Real Power')
            rpLayout = QtWidgets.QVBoxLayout(rpFrame)
            self.rpMatrix = MatrixWidget()
            rpLayout.addWidget(self.rpMatrix)
            self.layout.addWidget(rpFrame)

            ipFrame = QtWidgets.QGroupBox('Imaginary Power')
            ipLayout = QtWidgets.QVBoxLayout(ipFrame)
            self.ipMatrix = MatrixWidget()
            ipLayout.addWidget(self.ipMatrix)
            self.layout.addWidget(ipFrame)


            self.layout.addStretch()


class WaveAnalysis(QtWidgets.QFrame, WaveAnalysisUI):
    def __init__(self, spectra, window, parent=None):
        super(WaveAnalysis, self).__init__(parent)

        self.spectra = spectra
        self.window = window
        self.ui = WaveAnalysisUI()
        self.ui.setupUI(self, window)

        self.updateCalculations() # should add update button later


    def updateCalculations(self):
        print('updating')

        dstrs = [dd.currentText() for dd in self.ui.axesDropdowns]
        #print(dstrs)

        ffts = [self.spectra.getfft(dstr) for dstr in dstrs]
        print(len(ffts[0]))

        # needs start and end frequencies sliders prob (for now can just use whole spectra)
        # need to correct this. start and end points aren't exactly great
        fO = 0
        fE = len(ffts[0]) // 2
        k = fE - fO - 1
        counts = np.array(range(int(k))) + 1
        steps = 2 * counts - 1
        #print(steps)

        ffts = [fft[fO:fE*2] for fft in ffts]
        sqrs = [fft * fft for fft in ffts]

        deltaf = 2.0 / (self.spectra.maxN * self.spectra.maxN)

        ps = [sum([sq[i] + sq[i+1] for i in steps]) * deltaf for sq in sqrs]
        #x,y  x,z  y,z
        axisPairs = [(ffts[0],ffts[1]),(ffts[0],ffts[2]),(ffts[1],ffts[2])]

        cs = [sum([fft0[i]*fft1[i] + fft0[i+1]*fft1[i+1] for i in steps])*deltaf for fft0,fft1 in axisPairs]
        qs = [sum([fft0[i]*fft1[i+1] - fft0[i+1]*fft1[i] for i in steps])*deltaf for fft0,fft1 in axisPairs]

        realPower = [[ps[0], cs[0], cs[1]], [cs[0], ps[1], cs[2]], [cs[1], cs[2], ps[2]]]
        imagPower = [[0.0, -qs[0], -qs[1]], [qs[0], 0.0, -qs[2]], [qs[1], qs[2], 0.0]]

        #for row in realPower:
        #    print(row)
        #for row in imagPower:
        #    print(row)

        self.ui.rpMatrix.setMatrix(realPower)
        self.ui.ipMatrix.setMatrix(imagPower)

#def calculateValues(self, K, freqList, fft, fO, fE):
#        # 2015 oct 19
#        # discrepancy freq[0:-1] multiplies data[1:], first data point is ignored?
#        # 2015 oct 23 refactor this too many sections
#        print("K", K, "FE - FO", fE - fO, "FE", fE, "FO", fO, "FREQLIST LEN", len(freqList))
#        if fE == fO:
#            print("waveAnalysis.calculateValues ERROR ")
#            return
#        k = fE - fO - 1
#        useFreq = freqList[fO:fE - 1]
#        counts = numpy.array(range(int(k))) + 1
#        # fix counts
#        steps = 2 * counts - 1
#        # use only fO:fE values
#        X, Y, Z, T = fft
#        print(len(X))
#        fftx = X[fO:fE * 2]
#        ffty = Y[fO:fE * 2]
#        fftz = Z[fO:fE * 2]
#        fftt = T[fO:fE * 2]
##       fftx,ffty,fftz,fftt = X[fO:fE],Y[fO,fE],Z[fO:fE], T[fO:fE]
##       fftx,ffty,fftz,fftt = fft
#        xSq, ySq, zSq, tSq = fftx * fftx, ffty * ffty, fftz * fftz, fftt * fftt
#        # real i , imaginary i + 1
#        deltaf = 2.0 / (self.spectra.npts * self.spectra.npts)
#        pxx = sum([xSq[i] + xSq[i + 1] for i in steps]) * deltaf
#        pyy = sum([ySq[i] + ySq[i + 1] for i in steps]) * deltaf
#        pzz = sum([zSq[i] + zSq[i + 1] for i in steps]) * deltaf
#        pbb = sum([tSq[i] + tSq[i + 1] for i in steps]) * deltaf
#        cxy = sum([fftx[i] * ffty[i] + fftx[i + 1] * ffty[i + 1] for i in steps]) * deltaf
#        cxz = sum([fftx[i] * fftz[i] + fftx[i + 1] * fftz[i + 1] for i in steps]) * deltaf
#        cyz = sum([ffty[i] * fftz[i] + ffty[i + 1] * fftz[i + 1] for i in steps]) * deltaf
#        qxy = sum([fftx[i] * ffty[i + 1] - fftx[i + 1] * ffty[i] for i in steps]) * deltaf
#        qxz = sum([fftx[i] * fftz[i + 1] - fftx[i + 1] * fftz[i] for i in steps]) * deltaf
#        qyz = sum([ffty[i] * fftz[i + 1] - ffty[i + 1] * fftz[i] for i in steps]) * deltaf
#        pi = [xSq[i] + xSq[i + 1] + ySq[i] + ySq[i + 1] + zSq[i] + zSq[i + 1] for i in steps]
#        print("pi", len(pi))
#        freqPi = pi * useFreq
#        self.wFreq = sum(freqPi) /sum(pi)
#        # real and imaginary Power Matrices
#        realPower = [[pxx, cxy, cxz], [cxy, pyy, cyz], [cxz, cyz, pzz]]
#        imagPower = [[0.0, -qxy, -qxz], [qxy, 0.0, -qyz], [qxz, qyz, 0.0]]
#        self.realPower = realPower
#        self.imagPower = imagPower
#        # Power Spectra Parameters
#        self.powSpeTra = powSpeTra = pxx + pyy + pzz
#        self.traAmp = sqrt(powSpeTra)
#        self.comPow = pbb
#        self.comAmp = sqrt(pbb)
#        self.comRat = pbb / powSpeTra
#        qqqd = self.avg[3]
#        qqqp = numpy.linalg.norm(self.avg[:-1])
#        self.fraCom = self.comAmp / qqqp if qqqd == 0 else self.comAmp / qqqd
#        self.fraTra = self.traAmp / qqqp if qqqd == 0 else self.traAmp / qqqd
#        # Joe Means Parameters
#        qqq = numpy.linalg.norm([qxy, qxz, qyz])
#        qkem = numpy.array([qyz / qqq, -qxz / qqq, qxy / qqq])  # propagation direction
#        avg = self.avg
#        qqqn = numpy.dot(qkem, avg[:-1])
#        if qqqn < 0:
#            qkem = qkem * -1
#            qqqn = numpy.dot(qkem, avg[:-1])
#        qqq = qqqn / qqqp
#        qtem = 57.295878 * acos(qqq)  # field angle
#        qqq = numpy.linalg.norm([cxy, cxz, cyz])
#        qdlm = numpy.array([cyz / qqq, cxz / qqq, cxy / qqq])
##       qqqn = qdlm[0] * avg[0] + qdlm[1] * avg[1] + qdlm[2] * avg[2]  # use dot
#        qqqn = numpy.dot(qdlm, avg[:-1])
#        qqq = qqqn / qqqp
#        qalm = 57.29578 * acos(qqq)
#        self.qkem = qkem
#        self.qtem = qtem
#        self.qdlm = qdlm
#        self.qalm = qalm

#        # means transformation matrix
#        yx = qkem[1] * avg[2] - qkem[2] * avg[1]
#        yy = qkem[2] * avg[0] - qkem[0] * avg[2]
#        yz = qkem[0] * avg[1] - qkem[1] * avg[0]
#        qyxyz = numpy.linalg.norm([yx, yy, yz])
#        yx = yx / qyxyz
#        yy = yy / qyxyz
#        yz = yz / qyxyz
#        xx = yy * qkem[2] - yz * qkem[1]
#        xy = yz * qkem[0] - yx * qkem[2]
#        xz = yx * qkem[1] - yy * qkem[0]
#        bmat = [[xx, yx, qkem[0]], [xy, yy, qkem[1]], [xz, yz, qkem[2]]]
#        rmat = numpy.transpose(realPower)
#        duhh, amat = numpy.linalg.eigh(rmat, UPLO="U")
#        self.thbk, self.thkk = getAngles(amat, avg, qkem)
#        # Transformed Values Spectral Matrices  (make routine (amat, {realPower, imagPower} out {r,i}pp
#        rpp = arpat(amat, realPower)
#        rpp = flip(rpp)
#        ipp = arpat(amat, imagPower)
#        ipp = flip(ipp)
#        rpmp = arpat(bmat, realPower)
#        ipmp = arpat(bmat, imagPower)
#        self.rpp = rpp
#        self.ipp = ipp
#        self.rpmp = rpmp
#        self.ipmp = ipmp
#        # born-wolf analysis
#        self.pp, self.ppm, self.elip, self.elipm, self.azim = bornWolf(rpp, ipp, rpmp, ipmp)
#        return