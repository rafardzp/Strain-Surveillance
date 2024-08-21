import numpy as np
from sklearn.metrics import pairwise_distances

class EfficientPIKE():
    def __init__(self, t):
        self.t = t

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, th=1e-6):
        '''
        Returns the PIKE kernel value k(X, Y) and, if desired, its gradient as well.

        Parameters
        ----------
        X_mz:   array of spectra positions (mz) with shape (n_samples_X, spectrum_length_X) which contains the positions of the peaks 
        X_i:    array of spectra intensities with shape (n_samples_X, spectrum_length_X) which contains  contains the intensities of the peaks
                Left argument of the returned kernel k(X, Y)
        Y_mz:   array of spectra positions (mz) with shape (n_samples_Y, spectrum_length_X) which contains the positions of the peaks 
        Y_i:    array of spectra intensities with shape (n_samples_Y, spectrum_length_X) which contains the intensities of the peaks
                Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.

        Returns
        -------
        K:      array, shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y) if Y is provided
        '''
        if Y_mz is None and Y_i is None:
            Y_mz = X_mz
            Y_i = X_i
       
        K = np.zeros((X_mz.shape[0], Y_mz.shape[0]))

        positions_x = X_mz[0,:].reshape(-1, 1)
        positions_y = Y_mz[0,:].reshape(-1, 1)
        distances = pairwise_distances(
                positions_x,
                positions_y,
                metric='sqeuclidean'
            )
        distances = np.exp(-distances / (4 * self.t))
        d = np.where(distances[0] < th)[0][0]
        
        for i,x in enumerate(X_i.T):

            intensities_y = Y_i.T[:(i+d), :]
            di = distances[i, :(i+d)].reshape(-1, 1)
            prod = intensities_y * di
            x = np.broadcast_to(x, (np.minimum(i+d,  X_i.shape[1]), X_i.shape[0])).T
            P = np.matmul(x, prod)
            K += P 
        
        return K / (4 * self.t * np.pi)