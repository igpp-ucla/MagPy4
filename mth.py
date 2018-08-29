
import numpy as np
import multiprocessing as mp

try:
    import pycdf
except Exception as e:
    pass

class Mth:

    IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    STRING_PRECISION = 10
    AXES = ['X','Y','Z']
    i = [0, 1, 2]
    R2D = 57.29577951308232 # 1 radian is this many degrees

    def empty(): # return an empty 2D list in 3x3 matrix form
        return [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        # returns a copy of array a

    def copy(a):
        return [[a[0][0],a[0][1],a[0][2]],
                [a[1][0],a[1][1],a[1][2]],
                [a[2][0],a[2][1],a[2][2]]]

    # manual matrix mult with matrix list format
    def mult(a, b):
        N = Mth.empty()
        for r in Mth.i:
            for c in Mth.i:
                for i in Mth.i:
                    N[r][c] += a[r][i] * b[i][c]
        return N

    # converts float to string
    def formatNumber(num):
        n = round(num, Mth.STRING_PRECISION)
        #if n >= 10000 or n <= 0.0001: #not sure how to handle this for now
            #return f'{n:e}'
        return f'{n}'

    # matrix are stringified to use as keys in data table
    # DATADICT is dict with dicts for each dstr with (k, v) : (matrix str, modified data)
    # this is probably stupid but i dont know what im doing
    def matToString(m, p=STRING_PRECISION):
        return (f'''[{Mth.formatNumber(m[0][0])}]
                    [{Mth.formatNumber(m[0][1])}]
                    [{Mth.formatNumber(m[0][2])}]
                    [{Mth.formatNumber(m[1][0])}]
                    [{Mth.formatNumber(m[1][1])}]
                    [{Mth.formatNumber(m[1][2])}]
                    [{Mth.formatNumber(m[2][0])}]
                    [{Mth.formatNumber(m[2][1])}]
                    [{Mth.formatNumber(m[2][2])}]''')

    def identity():
        return Mth.matToString(Mth.IDENTITY)


    # mat is 2D list of label/lineEdits, m is 2D list of floats
    def setMatrix(mat, m):
        for i in Mth.i:
            for j in Mth.i:
                mat[i][j].setText(Mth.formatNumber(m[i][j]))
                mat[i][j].repaint() # mac doesnt repaint sometimes


    # mat is 2D list of label/lineEdits
    def getMatrix(mat):
        M = Mth.empty()
        for i in Mth.i:
            for j in Mth.i:
                s = mat[i][j].text()
                try:
                    f = float(s)
                except ValueError:
                    print(f'matrix has non-number at location {i},{j}')
                    f = 0.0
                M[i][j] = f
        return M

    def clamp(value, minimum, maximum):
        return max(min(value,maximum),minimum)

    def getSegmentsFromErrorsAndGaps(data, res, errorFlag, maxRes):
        if np.isnan(errorFlag):
            return Mth.getSegmentsFromTimeGaps(res, maxRes)
        elif errorFlag > 0:
            segments = np.where(np.logical_or(data >= errorFlag, res > maxRes))[0].tolist() 
        else:
            segments = np.where(np.logical_or(data <= errorFlag, res > maxRes))[0].tolist() 
        return Mth.__processSegmentList(segments, len(data))

    def getSegmentsFromErrors(data, errorFlag):
        if np.isnan(errorFlag):
            segments = []
        elif errorFlag > 0:
            segments = np.where(data >= errorFlag)[0].tolist()
        else:
            segments = np.where(data <= errorFlag)[0].tolist()
        return Mth.__processSegmentList(segments, len(data))

    def getSegmentsFromTimeGaps(res, maxRes):
        segments = np.where(res > maxRes)[0].tolist() 
        return Mth.__processSegmentList(segments, len(res))

    def __processSegmentList(segments, dataLen):
        segments.append(dataLen) # add one to end so last segment will be added (also if no errors)
        #print(f'SEGMENTS {len(segments)}')
        segList = []
        st = 0 #start index
        for seg in segments: # collect start and end range of each segment
            while st in segments:
                st += 1
            if st >= seg:
                continue
            segList.append((st,seg))
            st = seg + 1
        # returns empty list if data is pure errors
        return segList    

    # this smooths over data gaps, required for spectra analysis?
    # errors before first and after last or just extended from those points
    # errors between are lerped between nearest points
    # todo: try modifying using np.interp (prob faster)
    # PROBLEM: many files have gaps in the time but no error values to read in between
    # so this function cant register them as actual gaps in that case still!! 
    # so this function only detects errors really
    # could make separate function to detect actual gaps in time so we have option not to draw lines between huge gaps
    def interpolateErrors(origData, errorFlag):
        segs = Mth.getSegmentsFromErrors(origData, errorFlag)
        data = np.copy(origData)
        if len(segs) == 0: # data is pure errors
            return data

        # if first segment doesnt start at 0 
        # set data 0 - X to data at X
        first = segs[0][0]
        if first != 0: 
            data[:first] = data[first]

        # interate over the gaps in the segment list
        # this could prob be sped up somehow
        for si in range(len(segs) - 1):
            gO = segs[si][1] # start of gap
            gE = segs[si + 1][0] # end of gap
            gSize = gE - gO + 1 # gap size
            for i in range(gO,gE): # calculate gap values by lerping from start to end
                t = (i - gO + 1) / gSize
                data[i] = (1 - t) * data[gO - 1] + t * data[gE]

        # if last segment doesnt end with last index of data
        # then set data X - end based on X
        last = segs[-1][1]
        if last != len(data):
            data[last - 1:len(data)] = data[last - 1]

        return data


    #def cdfInternal(n):
    #    dt = 32.184   # tai - tt time?
    #    div = 10 ** 9
    #    d2tt2 = pycdf.lib.datetime_to_tt2000
    #    return d2tt2(n)/div-dt

    #def CDFEpochToTimeTicks(cdfEpoch):
    #    dt = 32.184   # tai - tt time?
    #    div = 10 ** 9
    #    d2tt2 = pycdf.lib.datetime_to_tt2000
    #    return d2tt2(cdfEpoch)/div-dt

    def CDFEpochToTimeTicks(cdfEpoch):
        # todo: add support for other epochs
        d2tt2 = pycdf.lib.datetime_to_tt2000
        num = len(cdfEpoch)
        arr = np.empty(num)

        #ttmJ2000 = 43167.8160001
        dt = 32.184   # tai - tt time?
        div = 10 ** 9

        rng = range(num)

        for i in rng:
            arr[i] = d2tt2(cdfEpoch[i]) / div - dt
        return arr