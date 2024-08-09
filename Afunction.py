import numpy as np
from astropy.convolution import convolve, convolve_fft


def Afunction(TF, x, shape=None): # todo: update docstring.
    """Describes the PSF function.

    Args:
        psf (numpy.ndarray): PSF matrix.
        x (numpy.ndarray): Image with which PSF needs to be convolved.

    Returns:
        numpy.ndarray: Convoluted version of image `x`.

    Note
    ----
    It uses the FFT version of the convolution to speed up the convolution process.
    In practice, I have found `convolve` to be much slower than `convolve_fft`.

    """
    x = x.reshape(shape)
    out = np.real(np.fft.ifftn(np.multiply(TF, np.fft.fftn(x))))
    out = out.flatten()
    return out

def zero_pad_arr_to_shape(a, shape):  # Credit: https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape
    """Function to zero-pad a PSF array to match a given shape. Useful when the PSF dimensions do not match the image dimensions exactly.
    """
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')


def Afunction_2d(psf, x, shape=None):
    """Describes the PSF function.

    Args:
        psf (numpy.ndarray): PSF matrix.
        x (numpy.ndarray): Image with which PSF needs to be convolved.

    Returns:
        numpy.ndarray: Convoluted version of image `x`.

    Note
    ----
    It uses the FFT version of the convolution to speed up the convolution process.
    In practice, I have found `convolve` to be much slower than `convolve_fft`.

    """
    x = x.reshape(shape)
    conv = convolve(x, psf, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
    return conv

def ATfunction_2d(psf, x, shape=None):
    """Describes the transposed PSF function.

    Args:
        psf (numpy.ndarray): PSF matrix.
        x (numpy.ndarray): Image with which PSF needs to be convolved.

    Returns:
        numpy.ndarray: Transpose-convoluted version of image `x`.

    Note
    ----
    It uses the FFT version of the convolution to speed up the convolution process.

    """
    x = x.reshape(shape)
    conv = convolve(x, psf.conj().T, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
    return conv
