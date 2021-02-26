from scipy import fft as fftlib
from scipy import signal
import numpy as np

class SpectraCalc():
    ''' Routines for computing power spectra and the coherence/phase
        between two signals
    '''
    def calc_fft(data):
        ''' Calculates the fast fourier transform of the given data

            Parameters:
            - data: Data to be passed to FFT algorithm (array_like)
        '''
        fft = fftlib.rfft(data)
        return fft

    def calc_freq(bw, n, res):
        ''' Calculates the frequency list for a spectra plot

            Parameters:
            - bw: Number of frequency bands to average over (int)
            - n: Number of data samples (int)
            - res: Data resolution in seconds (float)
        '''
        # Calculate the number of bands to skip at each end
        bw = SpectraCalc._round_bw(bw)
        half = int(bw/2)

        # Calculate the frequency for each band
        freq = np.fft.rfftfreq(n, d=res)
        high = len(freq) - half
        freq = freq[half+1:high]

        if len(freq) < 2:
            print ('Proposed plot invalid!\nFrequency list has < 2 values')
            return []

        return freq

    def calc_power(data, bw, res):
        ''' Calculate the power spectral density of a signal

            Parameters:
            - data: Fast fourier transform of the data (array_like)
            - bw: Number of frequency bands to average over (int)
            - res: Data resolution in seconds (float)
        '''
        # Calculate fast fourier transform
        fft = SpectraCalc.calc_fft(data)

        # Calculate the power from the fast fourier transform
        n = len(data)
        power = SpectraCalc.calc_power_fft(fft, bw, res, n)

        return power

    def calc_power_fft(fft, bw, res, n):
        ''' Calculate the power from the given fast
            fourier transform
        '''
        # Calculate parameters
        bw = SpectraCalc._round_bw(bw)
        c, half, nband, nfreq = SpectraCalc._get_power_vars(bw, n, res)

        # Calculate fft and skip first index
        fft = fft[1:]

        # Calculate power
        real = (fft.real ** 2)
        imag = (fft.imag ** 2)
        power = real + imag

        power = signal.oaconvolve(power, np.ones(bw), mode='valid')
        c = 2 * res / n
        power = power / bw * c

        return power

    def calc_coh_pha(data, bw, res):
        ''' Calculate the coherence and phase between two signals

            Parameters:
            - data: Data samples where each row is a different signal (array_like)
            - bw: Number of frequency bands to average over (int)
            - res: Data resolution in seconds (float)
        '''
        # Calculate fft
        fft = SpectraCalc.calc_fft(data)
        n = len(data[0])
        result = SpectraCalc.calc_coh_pha_fft(fft[0], fft[1], bw, res, n)
        return result

    def calc_coh_pha_fft(ffta, fftb, bw, res, n):
        ''' Calculate the coherence and phase between signals using their
            given Fast Fourier Transforms
        '''
        # Split fft in real/imaginary components
        a_real, a_imag = SpectraCalc._split_fft(ffta[1:])
        b_real, b_imag = SpectraCalc._split_fft(fftb[1:])

        # Pre-calculate cospectral and quasispectral matrix values
        c_aa = (a_real ** 2 + a_imag ** 2)
        c_bb = (b_real ** 2 + b_imag ** 2)

        c_ab = ((a_real*b_real) + (a_imag*b_imag))
        q_ab = -(((a_imag*b_real) - (b_imag*a_real)))

        # Adjust parameters
        bw = SpectraCalc._round_bw(bw)
        c, half, nband, nfreq = SpectraCalc._get_power_vars(bw, n, res)

        # Compute coherence and phase
        ## Average over matrix values using bw
        v = np.ones(bw)
        c_ab = signal.oaconvolve(c_ab, v, mode='valid', axes=0)
        c_aa = signal.oaconvolve(c_aa, v, mode='valid', axes=0)
        c_bb = signal.oaconvolve(c_bb, v, mode='valid', axes=0)
        q_ab = signal.oaconvolve(q_ab, v, mode='valid', axes=0)

        ## Calculate and convert to correct formats
        coh = ((c_ab ** 2) + (q_ab ** 2)) / (c_aa * c_bb)
        phase = np.arctan2(q_ab, c_ab)
        phase = np.rad2deg(phase)
        return (coh[:nfreq], phase[:nfreq])

    def calc_compress_power(data, bw, res):
        ''' Compute the compressional power of the signals in data
            where the rows represent the x, y, z signals
        '''
        mag = np.sqrt(np.sum(data ** 2, axis=0))
        power = SpectraCalc.calc_power(mag, bw, res)
        return power

    def calc_sum_powers(data, bw, res):
        ''' Compute the trace of the power of the signals in data
            where the rows represent the x, y, z signals
        '''
        rows = len(data)
        powers = []
        for i in range(rows):
            power = SpectraCalc.calc_power(data[i], bw, res)
            powers.append(power)

        sum_powers = np.sum(powers, axis=0)
        return sum_powers

    def calc_tranv_power(data, bw, res):
        ''' Compute the tranverse power of in data
            where the rows represent the x, y, z signals
        '''
        sum_powers = SpectraCalc.calc_sum_powers(data, bw, res)
        comp_power = SpectraCalc.calc_compress_power(data, bw, res)
        return np.abs(sum_powers - comp_power)

    def _round_bw(bw):
        ''' Rounds up bandwidth bw if it is not odd '''
        if bw % 2 == 0:
            bw += 1
        return bw

    def _get_power_vars(bw, n, res):
        ''' Calculate number of frequency bands and other
            commonly used parameters
        '''
        c = (2*res)/n
        half = int(bw/2.0)
        nband = n/2
        nfreq = int(nband - bw + 1)

        return c, half, nband, nfreq

    def _split_fft(fft):
        '''
            Splits FFT results into its real and imaginary parts
        '''
        real = fft.real
        imag = fft.imag
        return real, imag

class WaveCalc():
    ''' Routines for computing wave analysis values
    '''
    def calc_ellip(data, fft_param, method='svd'):
        ''' Calculate the ellipticity values for a wave

            Parameters:
            - data: Data samples where the rows correspond to Bx, By,
                    and Bz (array_like)
            - fft_param: A dictionary containing values for 
                'bandwidth' -> Number of frequency bands to average over (int)
                'resolution' -> Data resolution in seconds (float)
                'num_points' -> Number of data samples (int)
            - method: Method from ['svd', 'means', 'born-wolf'] (str)
        '''
        if method == 'svd':
            return WaveCalc._svd_ellip(data, fft_param)
        elif method == 'means':
            return WaveCalc._means_ellip(data, fft_param)
        else:
            return WaveCalc._bw_ellip(data, fft_param)

    def calc_prop_angle(data, fft_param, method='svd'):
        ''' Calculate the propagation angles for a wave

            Parameters:
            - data: Data samples where the rows correspond to Bx, By,
                    and Bz (array_like)
            - fft_param: A dictionary containing values for 
                'bandwidth' -> Number of frequency bands to average over (int)
                'resolution' -> Data resolution in seconds (float)
                'num_points' -> Number of data samples (int)
            - method: Method from ['svd', 'means', 'min-var'] (str)
        '''
        if method == 'svd':
            return WaveCalc._svd_prop_angle(data, fft_param)
        elif method == 'means':
            return WaveCalc._means_prop_angle(data, fft_param)
        else:
            return WaveCalc._min_var_prop_angle(data, fft_param)

    def calc_azimuth_angle(data, fft_param):
        '''Calculates the azimuth angle for a wave

            Parameters:
            - data: Data samples where the rows correspond to Bx, By,
                    and Bz (array_like)
            - fft_param: A dictionary containing values for 
                'bandwidth' -> Number of frequency bands to average over (int)
                'resolution' -> Data resolution in seconds (float)
                'num_points' -> Number of data samples (int)
        '''
        return WaveCalc._means_azim_angle(data, fft_param)

    def calc_power_trace(data, fft_param):
        ''' Computes the power spectra trace for a wave

            Parameters:
            - data: Data samples where the rows correspond to Bx, By,
                    and Bz (array_like)
            - fft_param: See definition for WaveCalc.calc_ellip
        '''
        bw = fft_param.get('bandwidth')
        res = fft_param.get('resolution')
        return SpectraCalc.calc_sum_powers(data, bw, res)

    def calc_compress_power(data, fft_param):
        ''' Computes the compressional power spectral density for a wave

            Parameters:
            - data: Data samples where the rows correspond to Bx, By,
                    and Bz (array_like)
            - fft_param: See definition for WaveCalc.calc_ellip
        '''
        bw = fft_param.get('bandwidth')
        res = fft_param.get('resolution')
        return SpectraCalc.calc_compress_power(data, bw, res)

    def _compute_mats(ffts):
        ''' Computes the covariance matrices from the given
            Fast Fourier Transforms for a set of signals

            Parameters:
            - ffts: Fast fourier transforms of each signal, where each
                row is a different signal (array_like)
        '''
        # Initialize array and lower triangular indices to iterate over
        mats = np.empty((len(ffts[0]), 3, 3), dtype=np.complex)
        xind, yind = np.tril_indices(3, 0, 3)

        # Compute fft * np.conj(fft) for each pair of signals
        for i, j in zip(xind, yind):
            value = ffts[i] * np.conj(ffts[j])
            mats[:,i,j] = value
            if i != j:
                mats[:,j,i] = np.conj(value)
        return mats

    def _get_mats(ffts, fft_param):
        ''' Calculate the averaged covariance matrices

            Parameters:
            - ffts: Fast fourier transforms of each signal, where each
                row is a different signal (array_like)
            - fft_param: A dictionary containing values for 
                'bandwidth' -> Number of frequency bands to average over (int)
                'resolution' -> Data resolution in seconds (float)
                'num_points' -> Number of data samples (int)
        '''
        # Extract fft parameters
        n = fft_param.get('num_points')
        bw = fft_param.get('bandwidth')
        res = fft_param.get('resolution')

        # Stack fft arrays (skip F[0])
        ffts = [fft[1:] for fft in ffts]

        # Map to (mx1) and (1xm) formats to compute matrices
        mats = WaveCalc._compute_mats(ffts)

        # Compute averaged matrices using convolution (overlapped method)
        v = [np.ones((3,3))]*bw
        avg_mats = signal.oaconvolve(mats, v, mode='valid', axes=0)

        # Scale by deltaf
        deltaf = (2. * res) / (n * bw)
        avg_mats *= deltaf

        return avg_mats

    def _new_basis(vec, other_vec):
        ''' Computes a new basis such that the z-axis is aligned
            with vec by computing the cross product between
            vec and other_vec, and then the cross product between
            this new vec and vec
        '''
        # Compute the cross product of vec and other_vec
        # to get a vector perpendicular to vec
        perp1 = np.cross(vec, other_vec)
        perp1 /= np.linalg.norm(perp1) # Normalize it

        # Repeat to get another vector orthogonal to perp1 and vec
        perp2 = np.cross(perp1, vec)
        perp2 /= np.linalg.norm(perp2)

        # Return the new basis of vectors stacked horizontally
        return np.vstack([perp2, perp1, vec])

    def _means_wave_vec_and_dot(mat, avg):
        ''' Computes the wave normal vector k from the given
            covariance matrix and the dot product between
            k and the average field

            Parameters:
            - mat: Cospectral matrix (array_like, np.complex)
            - avg: Background field (array_like)
        '''
        # Get the wave normal vector
        jyz = mat[1][2]
        jxz = mat[0][2]
        jxy = mat[0][1]
        vec = [jyz, -jxz, jxy]

        # Calculate ab coeff from the norm of the wave normal vec
        norm = np.linalg.norm(vec)
        vec = np.array(vec) / norm

        # Check if vec is right-handed or left-handed,
        # (right means vec dot avg > 0) and adjust sign accordingly
        dot = np.dot(vec, avg)
        if dot < 0:
            vec *= (-1)
            dot *= (-1)

        return vec, dot

    def _means_mat_and_vec(data, avg, fft_param):
        ''' Returns the covariance matrices, wave normal vectors,
            and dot product between the wave normal vectors and the
            avg background field

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - avg: Background field (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute fft of signals in data
        ffts = SpectraCalc.calc_fft(data)

        # Compute covariance matrices in complex form
        mats = WaveCalc._get_mats(ffts, fft_param)

        # Compute the wave normal vectors
        vecs = []
        dot_prods = []
        for t in range(0, len(mats)):
            mat = mats[t].imag

            # Compute the wave normal vector k and the dot product
            # between k and avg
            vec, dot = WaveCalc._means_wave_vec_and_dot(mat, avg)

            vecs.append(vec)
            dot_prods.append(dot)

        return (mats, vecs, dot_prods)

    def _means_change_of_basis(mat, vec, avg):
        ''' Computes a new basis such that the vector vec is the third vector
            and rotates the matrix mat such that the z-axis is aligned with vec
        '''
        # Compute a new basis using cross products;
        # The wave normal vector has norm = 1 and perp1 and perp2 were both 
        # normalized in _new_basis so we have an orthonormal set of vectors
        r = WaveCalc._new_basis(vec, avg)
        r_inv = r.T

        # Use the wave normal vec and perp1 and perp2 as the set of vectors
        # change the basis to by computing R*M*R^-1 (since the set is 
        # orthonormal, R inverse = R transpose)
        rotated_mat = np.matmul(np.matmul(r, mat), r_inv)

        return rotated_mat

    def _means_mat_trace_det(mat):
        ''' Compute the real trace and determinant of the given
            matrix using Means' method
        '''
        trace = mat.real[0][0] + mat.real[1][1]
        det = np.linalg.det(mat[:2][:,:2])
        return trace, det

    def _means_perc_polar(rotated_mat):
        ''' Compute the percent polarization from the matrix rotated
            using Means' method
        '''
        trace, det = WaveCalc._means_mat_trace_det(rotated_mat)
        fnum = 1 - (4*det.real) / (trace ** 2)
        if fnum <= 0:
            pp = 0
        else:
            pp = 100 * np.sqrt(fnum)
        return pp

    def _means_ellip_from_mat(rotated_mat):
        ''' Calculate the ellipticity using Means' method
            from the given rotated matrix
        '''
        # Compute the ellipticity
        imag_mat = rotated_mat.imag
        trace, det = WaveCalc._means_mat_trace_det(rotated_mat)

        intensity = (trace ** 2) - 4 * det
        if intensity <= 0:
            denom = 1
        else:
            denom = np.sqrt(intensity)
        value = 2 * imag_mat[0][1] / denom
        value = value.real

        ellip = np.tan(0.5*np.arcsin(value))
        return ellip

    def _means_ellip(data, fft_param):
        ''' Computes the ellipticity using Means' method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute background field vector
        avg = np.mean(data, axis=1)

        # Get covariance matrices and wave normal vectors
        mats, vecs, dot_prods = WaveCalc._means_mat_and_vec(data, avg, fft_param)

        # Rotate matrix into coordinate system with wave normal
        # vector as being an axis and then compute the ellipticity
        # from the rotate matrix
        ellip_vals = []
        for mat, vec in zip(mats, vecs):
            # Rotate matrix
            rotated_mat = WaveCalc._means_change_of_basis(mat, vec, avg)

            # Compute ellipticity from matrix
            ellip = WaveCalc._means_ellip_from_mat(rotated_mat)
            ellip_vals.append(ellip)

        return ellip_vals

    def _means_prop_angle(data, fft_param):
        ''' Computes the propagation angle in degrees using Means' method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute background field vector
        avg = np.mean(data, axis=1)

        # Get covariance matrices and wave normal vectors
        mats, vecs, dot_prods = WaveCalc._means_mat_and_vec(data, avg, fft_param)

        # Calculate propagation angle from dot product between
        # wave normal vector and background field (avg)
        norm = np.linalg.norm(avg)
        cos_theta = np.array(dot_prods) / norm
        prop_angles = np.rad2deg(np.arccos(cos_theta))

        return prop_angles

    def _means_azim_angle_from_mat(rotated_mat):
        ''' Computes the azimuth angle from the matrix rotated
            using Means' method
        '''
        real_mat = rotated_mat.real
        imag_mat = rotated_mat.imag

        fnum = 2 * real_mat[0][1]
        difm = real_mat[0][0] - real_mat[1][1]
        angle = fnum / difm
        azim = 0.5 * np.rad2deg(np.arctan(angle))
        return azim

    def _means_azim_angle(data, fft_param):
        ''' Computes the azimuth angle using Means' method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute background field vector
        avg = np.mean(data, axis=1)

        # Get covariance matrices and wave normal vectors
        info = WaveCalc._means_mat_and_vec(data, avg, fft_param)
        mats, vecs, dot_prods = info

        # Rotate matrix into coordinate system with wave normal
        # vector as being an axis and then compute azimuth angle
        angles = []
        for mat, vec in zip(mats, vecs):
            # Rotate matrix
            rotated_mat = WaveCalc._means_change_of_basis(mat, vec, avg)

            # Calculate azimuth angle
            azim = WaveCalc._means_azim_angle_from_mat(rotated_mat)

            angles.append(azim)

        return angles

    def _bw_ellip(data, fft_param):
        ''' Computes the ellipticity using the Born-Wolf method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute fft for each signal
        ffts = SpectraCalc.calc_fft(data)

        # Compute covariance matrices
        mats = WaveCalc._get_mats(ffts, fft_param)

        # Compute ellipticity with Born Wolf method
        ellip_vals = []
        for mat in mats:
            # Compute eigenvalues and eigenvectors
            evs, basis = np.linalg.eigh(mat.real, UPLO='U')
            inv_basis = basis[::,::-1]
            basis = inv_basis.T

            # Change basis of covariance matrix
            rot_mat = np.matmul(np.matmul(basis, mat), inv_basis)

            # Compute the polarization parameters using Mean's method
            ellip = WaveCalc._means_ellip_from_mat(rot_mat)
            ellip_vals.append(abs(ellip))

        return ellip_vals

    def _min_var_prop_angle(data, fft_param):
        ''' Computes the minimum variance propagation angle

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Compute background field vector
        avg = np.mean(data, axis=1)

        # Compute fft for each signal
        ffts = SpectraCalc.calc_fft(data)

        # Compute covariance matrices
        mats = WaveCalc._get_mats(ffts, fft_param)

        angles = []
        for mat in mats:
            # Compute eigenvalues and eigenvectors
            evs, eigenvectors = np.linalg.eigh(mat.real, UPLO='U')
            eigenvectors = eigenvectors.T

            # Compute theta
            evn = eigenvectors[:,0]
            evi = eigenvectors[:,1]
            evx = eigenvectors[:,2]

            q = np.dot(avg, evn)
            if q < 0:
                evn = -1*evn
                evi = -1*evi
            if evx[2] < 0:
                evx = -1*evx
                evi = -1*evi

            evc = np.cross(evx, evi)
            q = np.dot(evc, evn)
            if q < 0:
                evi = -1*evi

            q = np.dot(evn, avg)
            vetm = np.dot(avg, avg)
            if vetm < 0:
                angle = 0
            else:
                norm = np.sqrt(vetm)
                angle = np.rad2deg(np.arccos(q/norm))

            angles.append(angle)

        return angles

    def _svd_ellip(data, fft_param):
        '''
            Computes the ellipticity using the Singular Value Decomposition method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters; See WaveCalc.calc_ellip definition (dict)
        '''
        # Calculate background field vector
        avg = np.mean(data, axis=1)

        # Calculate fft of signals in data
        ffts = SpectraCalc.calc_fft(data)

        # Compute covariance matrices
        mats = WaveCalc._get_mats(ffts, fft_param)

        # Compute matrix for rotating J such that z-axis is aligned with avg field
        avg /= np.linalg.norm(avg)
        new_basis = WaveCalc._new_basis(avg, [0,0,1])
        new_basis_inv = new_basis.T

        # Stack imaginary parts of spectral matrix to create a system of linear eq
        # and compute the ellipticity from its singular values
        systems = np.hstack([mats.real, mats.imag * -1])
        ellip_vals = []
        for system, mat in zip(systems, mats):
            # Calculate singular values
            sing_vals = np.linalg.svd(system, compute_uv=False)
            sing_vals.sort()

            # Calculate ellipticity from ratio of two singular values
            ellip = sing_vals[1] / sing_vals[2]

            # Rotate imaginary matrix so that z-axis is aligned with
            # background field avg
            mat = np.matmul(np.matmul(new_basis, mat), new_basis_inv)

            # Adjust sign based on imaginary matrix's xy component
            ellip *= np.sign(mat.imag[0][1])
            ellip_vals.append(ellip)

        return ellip_vals

    def _svd_prop_angle(data, fft_param):
        '''
            Computes the propagation angle using the Singular Value
            Decomposition method

            Parameters:
            - data: Data samples where each row is a separate signal (array_like)
            - fft_params: FFT parameters (dict)
        '''
        # Calculate average background field vector and normalize it
        avg = np.mean(data, axis=1)
        avg /= np.linalg.norm(avg)

        # Calculate fft of signals in data
        ffts = SpectraCalc.calc_fft(data)

        # Compute covariance matrices
        mats = WaveCalc._get_mats(ffts, fft_param)

        # Set up system of linear equations and compute prop angles
        systems = np.hstack([mats.real, mats.imag * -1])
        prop_angles = []
        for system in systems:
            # Get singular value decomposition for the matrix formed by
            # stacking the real and imaginary parts of spectral matrix
            umat, sing_vals, vmat = np.linalg.svd(system)

            # Wave normal vector corresponds to vmat's row that
            # corresponds to the minimum singular value
            vec = vmat[2]

            # Calculate the dot product between the background field and
            # the wave normal vector, then solve for the arccos to get the angle
            cos_theta = np.dot(vec, avg)
            theta = np.rad2deg(np.arccos(cos_theta))

            # Wrap value
            if theta > 90:
                theta = 180 - theta

            prop_angles.append(theta)

        return prop_angles

class SpecWave(WaveCalc):
    ''' Routines for computing additional wave analysis parameters
        in WaveAnalysis subwindow in Spectra tool
    '''
    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self._mat = None

    def _get_mat(self, ffts, fft_param):
        ''' Computes the covariance matrix from the FFTs of the signal

            Parameters:
            ffts - Fast fourier transforms of the data where each
                row is a different signal (array_like)
            fft_param - A dictionary of parameters 
                - num_points = Number of samples (int)
                - resolution = Data resolution in seconds (float)
                - freq_range = Range of frequencies to compute over (tuple of ints)
        '''
        if self._mat is not None:
            return self._mat

        # Extract fft parameters
        n = fft_param.get('num_points')
        res = fft_param.get('resolution')
        fft_min, fft_max = fft_param.get('freq_range')

        # Clip fft range
        ffts = np.vstack(ffts)
        ffts = ffts[:,fft_min:fft_max]

        # Map to (mx1) and (1xm) formats to compute matrices
        mats = SpecWave._compute_mats(ffts)

        # Sum over matrices and scale by deltaf
        deltaf = 2.0 / (n ** 2)
        summat = np.sum(mats, axis=0) * deltaf
        summat = np.array(summat, dtype=np.complex64)

        # Return single summed matrix
        self._mat = np.array([summat], dtype=np.complex)
        return self._mat

    def svd_params(self, ffts, params, avg):
        ''' Compute SVD method parameters '''
        # Get covariance matrix and set up system of linear equations
        mats = self._get_mat(ffts, params)
        system = np.hstack([mats.real, mats.imag * -1])[0]

        # Get singular value decomposition for the matrix formed by
        # stacking the real and imaginary parts of spectral matrix
        umat, sing_vals, vmat = np.linalg.svd(system)
        sing_vals.sort()

        # Calculate ellipticity from ratio of singular values
        ellip = sing_vals[1] / sing_vals[2]

        # Rotate matrix to get sign
        basis = SpecWave._new_basis(avg, [0, 0, 1])
        mat = np.matmul(np.matmul(basis, mats[0]), basis.T)
        ellip *= np.sign(mat.imag[0][1])

        # Wave normal vector corresponds to vmat's row that
        # corresponds to the minimum singular value
        vec = vmat[2]

        # Calculate the dot product between the background field and
        # the wave normal vector, then solve for the arccos to get the angle
        cos_theta = np.dot(vec, avg)
        theta = np.rad2deg(np.arccos(cos_theta))

        # Wrap value
        if theta > 90:
            theta = 180 - theta

        results = {
            'svd_ellip' : ellip, 
            'svd_prop_angle' : theta,
            'svd_k' : vec
        }
        return results

    def bw_params(self, ffts, params, avg):
        ''' Computes Born-Wolf parameters '''
        # Get the covariance matrix
        mat = self._get_mat(ffts, params)[0]

        # Compute the eigenvectors and rotate matrix into
        # its new basis
        evs, eigen_vecs = np.linalg.eigh(mat.real, UPLO='U')
        basis = eigen_vecs[::,::-1].T
        rot_mat = np.matmul(np.matmul(basis, mat), basis.T)

        # Compute ellipticity and percent polarization
        ellip = SpecWave._means_ellip_from_mat(rot_mat)

        pp = SpecWave._means_perc_polar(rot_mat)

        results = {
            'bw_eigenvectors' : eigen_vecs,
            'bw_eigenvalues' : evs,
            'bw_ellip' : ellip,
            'bw_perc_polar' : pp,
            'bw_transf_pow' : rot_mat,
        }

        return results

    def lin_var_params(self, ffts, params, avg):
        ''' Computes parameters related to linear variance calculations '''
        # Get the covariance matrix
        mat = self._get_mat(ffts, params)[0]
        real_mat = mat.real

        # Get linear variance vector
        indices = [(0,1), (0,2), (1,2)] # Jxy, Jxz, Jyz
        vec = [real_mat[i][j] for (i, j) in indices][::-1]
        vec = np.array(vec) / np.linalg.norm(vec)

        # Find angle between linear variance vector and background field
        avg_mag = np.linalg.norm(avg)
        arccos = np.dot(vec, avg) / avg_mag
        angle = np.rad2deg(np.arccos(arccos))

        params = {
            'lin_var_vec' : vec,
            'lin_var_angle' : angle,
        }

        return params

    def means_params(self, ffts, params, avg):
        ''' Compute Joe Means parameters '''
        # Get the covariance matrix
        mat = self._get_mat(ffts, params)
        mat = mat[0]

        # Compute rotated matrix, ellipticity, and wave normal vector
        k, dot = SpecWave._means_wave_vec_and_dot(mat.imag, avg)
        rot_mat = SpecWave._means_change_of_basis(mat, k, avg)
        ellip = SpecWave._means_ellip_from_mat(rot_mat)

        # Compute propagation angle
        norm = np.linalg.norm(avg)
        cos_theta = dot / norm
        prop_angle = np.rad2deg(np.arccos(cos_theta))

        # Compute azimuth angle
        azimuth = SpecWave._means_azim_angle_from_mat(rot_mat)

        # Compute percent polarization
        pp = SpecWave._means_perc_polar(rot_mat)

        results = {
            'means_ellip' : ellip,
            'means_prop_angle' : prop_angle,
            'means_k' : k,
            'means_perc_polar' : pp,
            'means_transf_mat' : rot_mat,
            'means_azim_angle' : azimuth,
        }
        return results

    def get_params(self, ffts, fft_params, avg):
        ''' Computes all parameters seen in WaveAnalysis subwindow
            of Spectra tool
        '''
        # Reset computed matrix
        self.set_defaults()

        # Get parameters from every function
        params = {}
        funcs = [self.svd_params, self.means_params, self.bw_params,
            self.lin_var_params]

        for func in funcs:
            params.update(func(ffts, fft_params, avg))

        # Update Born-Wolf ellipticity using sign from Means method
        # rotated matrix
        rot_mat = params['means_transf_mat']
        sign = np.sign(rot_mat[0][1].imag)
        params['bw_ellip'] *= sign

        # Add in cospectral and quasispectral matrix
        params['sum_mat'] = self._get_mat(ffts, fft_params)[0]
        return params