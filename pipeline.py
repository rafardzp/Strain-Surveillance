import numpy as np
import pandas as pd
from normalization import ScaleNormalizer
from e_pike import EfficientPIKE

class KernelPipeline():
    '''
    A pipeline for processing MALDI data with different peak removal strategies and computing the E-PIKE.
    
    Attributes
    ----------
    peak_removal : str or None
        The strategy for peak removal. Options are None, 'masked', or 'spr'.
    common_peaks : array-like or None
        The common peak indicies in Da used when `peak_removal` is not None.
    '''
    def __init__(self, peak_removal=None, common_peaks=None):
        self.peak_removal = peak_removal
        self.common_peaks = common_peaks

        if self.peak_removal is not None and self.common_peaks is None:
            raise ValueError("`common_peaks` must be provided when `peak_removal` is not set to 'None'")

        # Strategies
        self.processing_mode = {
            None: self._no_peak_removal,
            'masked': self._masked_peaks,
            'spr': self._spr
        }

        # Access variables
        self.binned_data = None
        self.norm_data = None
        self.denoised_data = None
        self.scaled_data = None
        self.prepared_data = None
        self.kernel = None
        self.norm_kernel = None
        self.Kspr = None
        self.scaler = None
        self.Xcp = None
        self.norm_Xcp = None
        self.scaled_Xcp = None
        self.prepared_Xcp = None
        self.K = None
        self.Kcp = None

    def _bin_data(self, data, bin_size):
        '''
        Bins the input MALDI data by averaging over specified bin sizes.

        Parameters
        ----------
        data : DataFrame
            The MALDI data to be binned.
        bin_size : int
            The size of the bins used to aggregate the data.

        Returns
        -------
        binned_data : DataFrame
            The binned MALDI data.
        '''
        bins = []
        for i in range(0, data.shape[1], bin_size):      
            bin_mean = data.iloc[:, i:i+bin_size].mean(axis=1)
            bins.append(bin_mean)

        binned_data = pd.DataFrame(bins).T

        return binned_data
    
    def _binned_normalization(self, data):
        '''
        Normalizes binned MALDI data so that the sum of all bins is 1.

        Parameters
        ----------
        data : DataFrame
            The binned MALDI data to be normalized.

        Returns
        -------
        data_norm : DataFrame
            The normalized MALDI data.
        '''
        data_norm = data.div(data.sum(axis=1), axis=0)

        return data_norm
    
    def _noise_removal(self, data):
        '''
        Removes noise from the MALDI data by setting values below a threshold to zero.

        Parameters
        ----------
        data : DataFrame
            The MALDI data from which noise is to be removed.

        Returns
        -------
        denoised_data : DataFrame
            The denoised MALDI data.
        '''
        mean_data = data.values.flatten().mean() 
        std_data = data.values.flatten().std()
        denoised_data = data.applymap(lambda x: 0 if x < mean_data - std_data else x)
        
        return denoised_data
    
    def _scale_normalization(self, data, return_scaler=False, scaler=None):
        '''
        Scales the normalized MALDI data using a scale normalization technique.

        Parameters
        ----------
        data : DataFrame
            The MALDI data to be scaled.
        return_scaler : bool, optional
            Whether to return the scaler object used for scaling. Default is False.
        scaler : Scaler object, optional
            A specific scaler instance to use for scaling. If None, a new `ScaleNormalizer` is created.

        Returns
        -------
        scaled_data : DataFrame
            The scaled MALDI data.
        scaler : Scaler object, optional
            The scaler object used for scaling, returned if `return_scaler` is True.
        '''
        if scaler is None:
            scaler = ScaleNormalizer()
        scaled_data = pd.DataFrame(scaler.fit_transform(data.to_numpy()))

        if return_scaler:
            return scaled_data, scaler
        
        return scaled_data

    def _prepare_data_PIKE(self, data):
        '''
        Prepares the MALDI data for PIKE kernel computation by converting it into the required format.

        Parameters
        ----------
        data : DataFrame
            The MALDI data to be prepared.

        Returns
        -------
        x : ndarray
            The prepared data in the format required for PIKE kernel computation (samples, positions, intensities).
        '''
        x = (data).to_numpy()
        x = np.array([np.stack((data.columns.to_list(), x[i])) for i in range(data.shape[0])])   
        x = np.transpose(x.astype(float), axes=(0,2,1))

        return x
    
    def _cosine_normalization(self, kernel):
        '''
        Normalizes a kernel matrix using cosine normalization.

        Parameters
        ----------
        kernel : ndarray
            The kernel matrix to be normalized.

        Returns
        -------
        norm_kernel : ndarray
            The cosine-normalized kernel matrix.
        '''
        diag_K = np.diag(kernel)
        outer_diag_K = np.sqrt(np.outer(diag_K, diag_K))
        norm_kernel = kernel / outer_diag_K

        return norm_kernel
    
    def _mask_peaks(self, data, peaks, masking_window):
        '''
        Masks peaks in the MALDI data within a specified window around given peak positions.

        Parameters
        ----------
        data : DataFrame
            The MALDI data in which peaks are to be masked.
        peaks : array-like
            The positions of the peaks to be masked.
        masking_window : int
            The size of the window around each peak position to mask.

        Returns
        -------
        masked_data : DataFrame
            The MALDI data with specified peaks masked.
        '''
        masked_data = data.copy()
        for idx in peaks:
            start_idx = max(0, idx - masking_window)
            end_idx = min(data.shape[1] - 1, idx + masking_window)
                
            columns_to_flatten = [col for col in masked_data.columns if start_idx <= int(col) <= end_idx]
            masked_data.loc[:, columns_to_flatten] = 0
        
        return masked_data
    
    def _create_common_peaks_MALDI(self, binned_data, peaks, bin_size, mz_range):
        '''
        Creates a common peaks matrix from binned MALDI data for use in SPR kernel computation.

        Parameters
        ----------
        binned_data : DataFrame
            The binned MALDI data.
        peaks : array-like
            The positions of the common peaks.
        bin_size : int
            The bin size used for data binning.
        mz_range : array-like
            The range of mz values in Da.

        Returns
        -------
        Xcp : DataFrame
            The common peaks matrix.
        '''
        bin_indices = (peaks - np.min(mz_range)) // bin_size
        selected_bins = binned_data.iloc[:, bin_indices]
        median_intensities = selected_bins.median(axis=0).to_numpy().flatten()
    
        column_names = [str(i) for i in range(0, int(np.ceil(len(mz_range) / bin_size)))]
        Xcp = pd.DataFrame(0, index=[0], columns=column_names)
        for i, col in enumerate(bin_indices):
            Xcp[str(col)] = median_intensities[i]

        return Xcp

    def __call__(self, maldi_data, bin_size=3, t=4, th=1e-6, masking_window=30):
        '''
        Executes the pipeline on the provided MALDI data with the specified parameters.

        Parameters
        ----------
        maldi_data : DataFrame
            The MALDI data to process.
        bin_size : int, optional
            The bin size to use for data binning. Default is 3.
        t : int, optional
            The smoothing factor parameter for the PIKE kernel. Default is 4.
        th : float, optional
            The threshold for the distance optimization in the E-PIKE kernel. Default is 1e-6.
        masking_window : int, optional
            The window size for masking peaks in Da. Default is 30.

        Returns
        -------
        norm_kernel : ndarray
            The normalized kernel matrix.
        '''
        return self.processing_mode[self.peak_removal](maldi_data, bin_size, t, th, masking_window)

    def _no_peak_removal(self, maldi_data, bin_size, t, th, masking_window):
        self.binned_data = self._bin_data(maldi_data, bin_size)
        self.norm_data = self._binned_normalization(self.binned_data)
        self.denoised_data = self._noise_removal(self.norm_data)
        self.scaled_data = self._scale_normalization(self.denoised_data)
        self.prepared_data = self._prepare_data_PIKE(self.scaled_data)

        epike = EfficientPIKE(t=t)
        self.kernel = epike(self.prepared_data[:,:,0], self.prepared_data[:,:,1], th=th)

        self.norm_kernel = self._cosine_normalization(self.kernel)
        return self.norm_kernel

    def _masked_peaks(self, maldi_data, bin_size, t, th, masking_window):
        self.masked_data = self._mask_peaks(maldi_data, self.common_peaks, masking_window)
        self.binned_data = self._bin_data(self.masked_data, bin_size)
        self.norm_data = self._binned_normalization(self.binned_data)
        self.denoised_data = self._noise_removal(self.norm_data)
        self.scaled_data = self._scale_normalization(self.denoised_data)
        self.prepared_data = self._prepare_data_PIKE(self.scaled_data)

        epike = EfficientPIKE(t=t)
        self.kernel = epike(self.prepared_data[:,:,0], self.prepared_data[:,:,1], th=th)

        self.norm_kernel = self._cosine_normalization(self.kernel)
        return self.norm_kernel

    def _spr(self, maldi_data, bin_size, t, th, masking_window):
        # Knospr
        self.binned_data = self._bin_data(maldi_data, bin_size)
        self.norm_data = self._binned_normalization(self.binned_data)
        self.denoised_data = self._noise_removal(self.norm_data)
        self.scaled_data, scaler = self._scale_normalization(self.denoised_data, return_scaler=True)
        self.prepared_data = self._prepare_data_PIKE(self.scaled_data)

        epike = EfficientPIKE(t=t)
        self.kernel = epike(self.prepared_data[:,:,0], self.prepared_data[:,:,1], th=th)

        # Xcp
        mz_range = [int(col) for col in maldi_data.columns]
        self.Xcp = self._create_common_peaks_MALDI(self.binned_data, self.common_peaks, bin_size, mz_range)
        self.norm_Xcp = self._binned_normalization(self.Xcp)
        self.scaled_Xcp = self._scale_normalization(self.norm_Xcp, scaler=scaler)
        self.prepared_Xcp = self._prepare_data_PIKE(self.scaled_Xcp)

        # Compute K
        self.K = epike(self.prepared_data[:,:,0], self.prepared_data[:,:,1], 
                       self.prepared_Xcp[:,:,0], self.prepared_Xcp[:,:,1], th=th)
        
        # Compute Kcp
        self.Kcp = epike(self.prepared_Xcp[:,:,0], self.prepared_Xcp[:,:,1], th=th)

        # Remove peaks in kernel space
        ones_n = np.ones((1, self.kernel.shape[0]))
        ones_nn = np.ones_like(self.kernel)
        self.Kspr = self.kernel - self.K @ ones_n - ones_n.T @ self.K.T + self.Kcp * ones_nn

        self.norm_Kspr = self._cosine_normalization(self.Kspr)
        return self.norm_Kspr
    
    def get_intermediate_matrices(self):
        return {
            'binned_data': self.binned_data,
            'norm_data': self.norm_data,
            'denoised_data': self.denoised_data,
            'scaled_data': self.scaled_data,
            'prepared_data': self.prepared_data,
            'kernel': self.kernel,
            'norm_kernel': self.norm_kernel,
            'Kspr': self.Kspr,
            'scaler': self.scaler,
            'Xcp': self.Xcp,
            'norm_Xcp': self.norm_Xcp,
            'scaled_Xcp': self.scaled_Xcp,
            'prepared_Xcp': self.prepared_Xcp,
            'K': self.K,
            'Kcp': self.Kcp
        }


