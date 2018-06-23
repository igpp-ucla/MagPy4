
class Mth:

    IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    STRING_PRECISION = 10
    AXES = ['X','Y','Z']
    i = [0, 1, 2]

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