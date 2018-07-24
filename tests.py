import numpy as np
from mth import Mth

class Tests:

    def test(name, result, expected):
        if result == expected:
            print(f'passed test \'{name}\'')
        else:
            print(f'failed test \'{name}\'')
            print(f'    expected: {expected}, got: {result}')

    def runTests():
        #runTest = lambda i,x,y: print(f'test {i} {"passed" if x==y else "failed"}')

        test = Tests.test

        t = [0,1,2,3,4,6,7,8,9]
        res = np.diff(t)
        res = np.append(res, res[-1])

        test('SegsFromTimeGaps_1', Mth.getSegmentsFromTimeGaps(res, 1), [(0,4),(5,9)])
        test('SegsFromTimeGaps_2', Mth.getSegmentsFromTimeGaps(res, 2), [(0,9)])

        t = [0,1,2,5,6,10,11,20]
        res = np.diff(t)
        res = np.append(res, res[-1])
        test('SegsFromTimeGaps_3', Mth.getSegmentsFromTimeGaps(res, 2), [(0,2),(3,4),(5,6)]) # kinda weird not sure if should include isolated singles technically?

        # maybe this failure is why renders correctly without +1 to end of each seg. there is a difference between resolution array and data array with errors which makes sense..
        test('SegsFromErrors_1', Mth.getSegmentsFromErrors(np.array([0,1,2,1e10,1,1e10,1e10]), 1e9), [(0,2),(4,4)])


