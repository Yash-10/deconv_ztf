# Taken from https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/restoration/deconvolution.py#L361-L421

import numpy as np
from scipy.signal import convolve
from functools import partial
from timeit import default_timer as timer

from Afunction import Afunction_2d, ATfunction_2d

new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}

def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)

def get_damped_rl_objective(I, D, T=2):
    return (-2 / (T ** 2)) * (D * np.log(I / D) - I + D)

def richardson_lucy(
    image, psf, bkg, num_iter=50, clip=False, filter_epsilon=None,
    tol=1e-4, flux=None, damped=True, T=3
):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be n-dimensional).
    psf : ndarray
       The point spread function.
    bkg: ndarray
        2-D background level. Array must be same size as `image`.
    num_iter : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    filter_epsilon: float, optional
       Value below which intermediate results become 0 to avoid division
       by small numbers.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    t0 = timer()  # Start clock timer.

    float_type = _supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    # im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    im_deconv = image.copy()
    # psf_mirror = np.flip(psf)

    # Small regularization parameter used to avoid 0 divisions
    eps = 1e-12

    # Compute objecive function value.
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e9 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        errmsg = f"\n\tsum(psf) - 1. = {checkPSF}, tolerance = {tolCheckPSF}"
        raise ValueError(f'PSF is not normalized! Provide a normalized PSF! {errmsg}')

    _shape = image.shape

    A = partial(Afunction_2d, psf=psf, shape=_shape)
    AT = partial(ATfunction_2d, psf=psf, shape=_shape)

    times = np.zeros(num_iter + 1)
    times[0] = 0

    if flux is None:
        flux = np.sum(image - bkg)
    x_tf = A(x=im_deconv)
    x_tf = x_tf.reshape(image.shape)
    den = x_tf + bkg
    temp = np.divide(image, den)
    fv = np.sum(np.multiply(image, np.log(temp))) + np.sum(x_tf) - flux
    if damped:
        N = 10
        _fv_damped = get_damped_rl_objective(image, den, T=T)
        _fv_damped = _fv_damped if _fv_damped >= 1 else (N-1)/(N+1) * (1 - _fv_damped**(N+1)) + _fv_damped ** N
        fv = np.sum(_fv_damped)

    M = 1
    Fold = -1e30 * np.ones(M)

    iter_ = 1

    _shape = image.shape
    A = partial(Afunction_2d, psf=psf, shape=_shape)

    for _ in range(num_iter):
        prev_x = im_deconv.copy()
        # conv = convolve(im_deconv, psf, mode='same') + eps + bkg
        conv = A(x=im_deconv).reshape(image.shape) + bkg
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        # im_deconv *= convolve(relative_blur, psf_mirror, mode='same')
        im_deconv *= AT(x=relative_blur).reshape(image.shape)

        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv
        x_tf = A(x=im_deconv)
        x_tf = x_tf.reshape(image.shape)
        den = x_tf + bkg
        temp = np.divide(image, den)
        fv = np.sum(np.multiply(image, np.log(temp))) + np.sum(x_tf) - flux
        if damped:
            _fv_damped = get_damped_rl_objective(image, den, T=T)
            _fv_damped = _fv_damped if _fv_damped >= 1 else (N-1)/(N+1) * (1 - _fv_damped**(N+1)) + _fv_damped ** N
            fv = np.sum(_fv_damped)

        reldecrease = (Fold[M-1]-fv) / fv
        loop = reldecrease > tol and reldecrease >= 0

        iter_ += 1
        times[iter_-1] = timer() - t0

        if not loop:
            im_deconv = prev_x
            break

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    times = times[0:iter_]
    iter_ = iter_ - 1

    return im_deconv, iter_, times
