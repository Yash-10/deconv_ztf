import math
import os
# import sys
import errno
import glob
import logging
import numpy as np
import pandas as pd
from functools import partial
# Use default_timer instead of timeit.timeit: Reasons here: https://stackoverflow.com/a/25823885
from timeit import default_timer as timer
import argparse

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

# from photutils.background import Background2D, MedianBackground, MeanBackground, StdBackgroundRMS

# from astropy.nddata import Cutout2D
# from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.wcs import WCS
# from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, convolve_fft
# from photutils.segmentation import detect_threshold, detect_sources, make_source_mask, SegmentationImage
# from astropy.convolution import interpolate_replace_nans

import matplotlib.pyplot as plt

from flux_conserve_proj import projectDF
# from utils import source_info, scale_psf, artificial_sky_background, create_subdivisions, reconstruct_full_image_from_patches, reconstruct_full_image_from_patches_original
from constants import DEFAULT_PARAMS, CAT_COLUMNS

from Afunction import Afunction_2d, ATfunction_2d, Afunction, zero_pad_arr_to_shape


def sgp(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, save=False, obj=None, verbose=True, flux=None,
    scale_data=True, errflag=False, tol_convergence=1e-4
):
    # TODO: Update docstring up to date.
    """Perform the SGP algorithm on a single star stamp.

    Args:
        gn (_type_): Observed star cutout image.
        psf (_type_): PSF matrix.
        bkg (_type_): Background level. This code only allows a 2D background level. If you have a single float value, you need to fill a 2D array with the background value with shape same as `gn`.
        init_recon (int, optional): Initialization for the reconstructed image.
            Either 0, 1, 2, or 3. Defaults to 0.
        proj_type (int, optional): Type of projection during the iteration.
            Either 0 or 1. Defaults to 0.
        stop_criterion (int, optional): Choice of rule to stop iteration.
            Either 1, 2, 3, or 4. Defaults to 0.
        MAXIT (int, optional): Maximum no. of iterations. Defaults to 500.
        gamma (_type_, optional): Linesearch penalty parameter. Defaults to 1e-4.
        beta (float, optional): Linesearch back-tracking/scaling parameter (used in the backtracking loop). Defaults to 0.4.
        alpha (float, optional): Initial value for alpha, the step length. Defaults to 1.3.
            This value is updated during the iterations.
        alpha_min (_type_, optional): alpha lower bound for Barzilai-Borwein' steplength. Defaults to 1e-5.
        alpha_max (_type_, optional): alpha upper bound for Barzilai-Borwein' steplength. Defaults to 1e5.
        M_alpha (int, optional): Memory length for `alphabb2`. Defaults to 3.
        tau (float, optional): Alternating parameter.. Defaults to 0.5.
        M (int, optional): Non-monotone linear search memory (M = 1 means monotone search). Defaults to 1.
        max_projs (int, optional): Maximum no. of iterations for the flux conservation procedure. Defaults to 1000.
        save (bool, optional): Whether to save the reconstructed image or not. Defaults to True.
        verbose (bool, optional): Controls screen verbosity. Defaults to True.
        flux (_type_, optional): Precomputed flux of the object inside `gn`. Defaults to None.
        scale_data (bool, optional): Whether to scale `gn`, 'bkg`, and `x` (the reconstructed image) before applying SGP. Defaults to False.

        tol_convergence: only used if stop_criterion == 2 or 3.

    Raises:
        OSError: _description_

    Returns:
        _type_: _description_

    Notes:
        == Porting from MATLAB to NumPy ==
        1. np.dot(a, b) is same as dot(a, b) ONLY for 1-D arrays.
        2. np.multiply(a, b) (or a * b) is same as a .* b
        3. If C = [1, 2, 3], then C[0:2] is same as C(1:2).
            In general, array[i:k] in NumPy is same as array(i+1:k) in MATLAB.
        4. x.conj().T in Numpy is the same as x' in matlab, where x is a two-dimensional array.

        * Afunction implementation is provided (similar to SGP-dec). A slightly different approach compared to SGP-dec is also commented out in the code.
        * See here: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for more details.

    """
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e9 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        errmsg = f"\n\tsum(psf) - 1. = {checkPSF}, tolerance = {tolCheckPSF}"
        raise ValueError(f'PSF is not normalized! Provide a normalized PSF! {errmsg}')

    logging.basicConfig(filename='sgp.log', level=logging.INFO, force=True)

    _shape = gn.shape

    A = partial(Afunction_2d, psf=psf, shape=_shape)
    AT = partial(ATfunction_2d, psf=psf, shape=_shape)

    ############ The below commented lines can be used to reproduce the original MATLAB code's behavior. ############
    # But I have personally found this to discard 10-15% sources in the deconvolved.
    # Thus, I am using the custom function defined above.
    # psf = zero_pad_arr_to_shape(psf, _shape)
    # TF = np.fft.fftn(np.fft.fftshift(psf))
    # CTF = np.conj(np.fft.fftn(np.fft.fftshift(psf)))
    # A = partial(Afunction, TF=TF, shape=_shape)
    # AT = partial(Afunction, TF=CTF, shape=_shape)
    #################################################################################################################

    t0 = timer()  # Start clock timer.

    # Initialization of reconstructed image.
    if init_recon == 0:
        x = np.zeros_like(gn)
    elif init_recon == 1:
        x = np.random.randn(*gn.shape)
    elif init_recon == 2:
        x = gn.copy()
    elif init_recon == 3:
        if flux is None:
            x = np.sum(gn - bkg) / gn.size * np.ones_like(gn)
        else:
            x = flux / gn.size * np.ones_like(gn)
    else:
        raise ValueError(f"init_recon = {init_recon} not recognized! Allowed values are 0 / 1 / 2 / 3")

    # Treat images as vectors.
    gn = gn.flatten()
    x = x.flatten()
    bkg = bkg.flatten()

    # Stop criterion settings.
    if stop_criterion == 1:
        tol = []
    elif stop_criterion == 2 or stop_criterion == 3:
        tol = tol_convergence
    elif stop_criterion == 4:
        tol = 1 + 1 / np.mean(gn)
    else:
        raise ValueError(f"stop_criterion = {stop_criterion} not recognized! Allowed values are 1 / 2 / 3 / 4")

    # Scaling
    if scale_data:
        scaling = np.max(gn)
        gn = gn / scaling
        bkg = bkg / scaling
        x = x / scaling
    else:
        scaling = 1.   # Scaling can have adverse effects on the flux in the final scaled output image, hence we do not scale.

    # Change null pixels of observed image.
    vmin = np.min(gn[gn > 0])
    eps = np.finfo(float).eps
    gn[gn <= 0] = vmin * eps * eps

    # Computations needed only once.
    N = gn.size
    if flux is None:
        flux = np.sum(gn - bkg)
    else:  # If flux is already provided, we need to scale it. This option is recommended.
        flux /= scaling  # Input a precomputed flux: this could be more accurate in some situations.

    iter_ = 1
    Valpha = alpha_max * np.ones(M_alpha)
    Fold = -1e30 * np.ones(M)
    Discr_coeff = 2 / N * scaling
    ONE = np.ones(N)

    # Projection type.
    pflag = proj_type  # Default: 0.

    # Setup directory to store reconstructed images.
    if save:
        dirname = "SGP_reconstructed_images/"
        try:
            os.mkdir(dirname)
            fits.writeto(f'{dirname}/orig.fits', gn.reshape(_shape))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError("Directory already exists!")
            pass

    discr = np.zeros(MAXIT + 1)
    times = np.zeros(MAXIT + 1)
    times[0] = 0

    if errflag and obj is None:
        raise ValueError("errflag was set to True but no ground-truth was passed.")

    if errflag:
        err = np.zeros(MAXIT + 1)
        obj = obj.flatten()
        obj = obj / scaling
        obj_sum = np.sum(obj * obj)

    # Start of SGP.
    # Project the initial point.
    if pflag == 0:
        x[x < 0] = 0
    elif pflag == 1:
        # Here an identity matrix is used for projecting the initial point, which means it is not a "scaled" gradient projection as of now.
        # Instead, it is a simple gradient projection without scaling (see equation 2 in https://kd.nsfc.gov.cn/paperDownload/ZD6608905.pdf, for example).
        x = projectDF(flux, x, np.ones_like(x), max_projs=max_projs)

    if errflag:
        e = x - obj
        err[0] = np.sqrt(np.sum(e * e) / obj_sum)

    # Compute objecive function value.
    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = ONE - AT(x=temp)
    # KL divergence.
    fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf) - flux

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + bkg)), AT(x=gn))

    if np.all(y < 0):
        # This is only to handle cases where no element in y has a positive value.
        # This generally happens when there is no detection in the image and flux can be negative.
        # Ideally if no sources are present, flux should be very close to zero,
        # but may not be zero and even become slightly negative sometimes. So this handles that.
        # Note that this value of 1e-10 is from the original SGP paper.
        # The variable bounds idea was from Prato et al., 2012 (who released the MATLAB SGP code).
        # That idea is still used unless all pixels in `y` are negative, in which case we fix the lower bound to 1e-10.
        X_low_bound = 1e-10
    else:
        X_low_bound = np.min(y[y > 0])

    X_upp_bound = np.max(y)
    if X_upp_bound / X_low_bound < 50:
        X_low_bound = X_low_bound / 10
        X_upp_bound = X_upp_bound * 10

    # Discrepancy.
    discr[0] = Discr_coeff * fv

    # Scaling matrix.
    if init_recon == 0:
        X = np.ones_like(x)
    else:
        X = x.copy()
        # Bounds
        X[X < X_low_bound] = X_low_bound
        X[X > X_upp_bound] = X_upp_bound

    if pflag == 1:
        D = np.divide(1, X)

    # Setup tolerance for main SGP iterations.
    if verbose:
        if stop_criterion == 2:
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 0 \n')
            tol = tol * tol
        elif stop_criterion == 3:
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | 0 \n')
        elif stop_criterion == 4:
            logging.info(f'it {iter_-1} D_k {discr[0]} \n')

    # Main loop.
    loop = True
    while loop:
        prev_x = x.copy()

        # Store alpha and obj func values.
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D, max_projs=max_projs)

        d = y - x

        # Backtracking loop for linearsearch.
        gd = np.dot(d, g)
        lam = 1  # `lam = 1` is a common choice, so we use it.

        fcontinue = 1
        d_tf = A(x=d)
        fr = max(Fold)

        while fcontinue:
            xplus = x + lam * d
            x_tf_try = x_tf + lam*d_tf
            den = x_tf_try + bkg

            temp = np.divide(gn, den)
            fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf_try) - flux

            if fv <= fr + gamma * lam * gd or lam < 1e-12:
                x = xplus.copy()
                xplus = None  # clear the variable.
                sk = lam*d
                x_tf = x_tf_try
                x_tf_try = None  # clear
                gtemp = ONE - AT(x=temp)

                yk = gtemp - g
                g = gtemp.copy()
                gtemp = None  # clear
                fcontinue = 0
            else:
                lam = lam * beta

        if fv >= fr and verbose:
            logging.warning("\tWarning, fv >= fr")

        # Update the scaling matrix and steplength
        X = x.copy()
        X[X < X_low_bound] = X_low_bound
        X[X > X_upp_bound] = X_upp_bound

        # assert all(np.isfinite(X)), "The scaling matrix violates either the lower or upper bound!"

        D = np.divide(1, X)
        sk2 = np.multiply(sk, D)
        yk2 = np.multiply(yk, X)
        bk = np.dot(sk2, yk)
        ck = np.dot(yk2, sk)
        if bk <= 0:
            alpha1 = min(10*alpha, alpha_max)
        else:
            alpha1bb = np.sum(np.dot(sk2, sk2)) / bk
            alpha1 = min(alpha_max, max(alpha_min, alpha1bb))
        if ck <= 0:
            alpha2 = min(10*alpha, alpha_max)
        else:
            alpha2bb = ck / np.sum(np.dot(yk2, yk2))
            alpha2 = min(alpha_max, max(alpha_min, alpha2bb))

        Valpha[M_alpha-1] = alpha2

        if iter_ <= 20:
            alpha = min(Valpha)
        elif alpha2/alpha1 < tau:
            alpha = min(Valpha)
            tau = tau * 0.9
        else:
            alpha = alpha1
            tau = tau * 1.1

        # Note: At this point, the original matlab code does: alpha = double(single(alpha)), which we do not do here. Don't know what's the meaning of that.

        iter_ += 1
        times[iter_-1] = timer() - t0
        discr[iter_-1] = Discr_coeff * fv

        if errflag:
            e = x - obj
            err[iter_] = np.sqrt(np.sum(e * e) / obj_sum)

        # Stop criterion.
        if stop_criterion == 1:
            logging.info(f'it {iter_-1} of  {MAXIT}\n')
        elif stop_criterion == 2:
            normstep = np.dot(sk, sk) / np.dot(x, x)
            loop = normstep > tol
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 {normstep} tol {tol}\n')
        elif stop_criterion == 3:
            reldecrease = (Fold[M-1]-fv) / fv
            loop = reldecrease > tol and reldecrease >= 0
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | {reldecrease} tol {tol}\n')
        elif stop_criterion == 4:
            loop = discr[iter_-1] > tol
            logging.info(f'it {iter_-1} D_k {discr[iter_-1]} tol {tol}\n')

        if iter_ > MAXIT:
            loop = False

        if save:
            filename = f'SGP_reconstructed_images/rec_{iter_-1}.fits'
            fits.writeto(filename, x.reshape(_shape), overwrite=True)
            # Residual image.
            res = np.divide(x - gn, np.sqrt(x))
            filename = f'SGP_reconstructed_images/res_{iter_-1}.fits'
            fits.writeto(filename, res.reshape(_shape), overwrite=True)

        if not loop:
            x = prev_x

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    if errflag:
        err = err[0:iter_]

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    return x, iter_, discr, times, err if errflag else None


def betaDiv(y, x, betaParam):
    """_summary_
    Args:
        y (_type_): _description_
        x (_type_): _description_
        betaParam (_type_): _description_
    Returns:
        _type_: _description_
    Note:
    - See an example implementation here: https://pytorch-nmf.readthedocs.io/en/stable/modules/metrics.html#torchnmf.metrics.beta_div
    """
    if betaParam == 0:
        return np.sum(x / y) - np.sum(np.log(x / y)) - x.size  # or y.size can also be used
    elif betaParam == 1:
        return np.sum(np.multiply(x, np.log(np.divide(x, y)))) - np.sum(x) + np.sum(y)
    else:
        scal = 1 / (betaParam * (betaParam - 1))
        return np.sum(scal*x**betaParam) + np.sum(scal*(betaParam-1)*y**betaParam) - np.sum(scal*betaParam*x*y**(betaParam-1))


# @njit
def betaDivDeriv(y, x, betaParam):  # Verified that the derivative is correct using pytorch backward() followed by .grad attribute checking.
    """ To get the derivative equation:
            from sympy import diff, symbols
            x, y, beta = symbols('x y beta')
            diff(betaDiv(y, x, beta), beta)
            then copy and paste the expression to return.
    Args:
        y (_type_): _description_
        x (_type_): _description_
        beta (_type_): _description_
    Returns:
        _type_: _description_
    Note:
    Comparing with PyTorch grad calculation:
    ```
    In [27]: x=torch.tensor([1,2, 4.5, 7.9, 1.5], requires_grad=True)
    In [28]: y=torch.tensor([9.3,2.5, 4.5, 7.9, 1.5], requires_grad=True)
    In [29]: f=betaDiv(y, x, betaParam)
    In [30]: betaParam=torch.tensor(1.5, requires_grad=True)
    In [31]: f=betaDiv(y, x, betaParam)
    In [32]: f.backward()
    In [33]: betaParam.grad
    Out[33]: tensor(24.6697)
    In [34]: betaDivDeriv(y, x, betaParam).sum()
    Out[34]: tensor(24.6697, grad_fn=<SumBackward0>)
    ```
    Indeed, both are same.
    """
    # t1 = (-(2*beta-1) / (beta**2-beta)**2) * (x**beta+(beta-1)*y**beta-beta*x*y**(beta-1))
    # t2 = (1 / (beta *(beta-1))) * (beta*x**(beta-1)+(beta-1)*beta*y**(beta-1)+y**beta-x*(y**(beta-1)+beta*(beta-1)*y**(beta-2)))
    # return t1+t2
    if betaParam == 0 or betaParam == 1:  # Special cases.
        return 0
    return -x*y**(betaParam - 1)*np.log(y)/(betaParam - 1) + x*y**(betaParam - 1)/(betaParam - 1)**2 + x**betaParam*np.log(x)/(betaParam*(betaParam - 1)) - x**betaParam/(betaParam*(betaParam - 1)**2) + y**betaParam*np.log(y)/betaParam - x**betaParam/(betaParam**2*(betaParam - 1)) - y**betaParam/betaParam**2


def betaDivDerivwrtY(AT, den_arg, gn_arg, betaParam):  # Verified and compared with the special case of KL divergence.
    return den_arg**(betaParam-1) - AT(x=gn_arg*den_arg**(betaParam-2))


def lr_schedule(init_lr, k, epoch):
    return init_lr * math.exp(-k * epoch)


################################################################################
################################ IMPORTANT NOTE ################################
# TODO: sgp_betaDiv is not maintained anymore and will be archived soon.
################################################################################
def sgp_betaDiv(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, save=False, obj=None, verbose=True, flux=None,
    scale_data=True, errflag=False, adapt_beta=True,
    betaParam=1.005, lr=1e-3, lr_exp_param=0.1, schedule_lr=False, tol_convergence=1e-4,
    use_original_SGP_Afunction=True
):
    # TODO: errflag and obj are just added. -- add implementation for them as well.
    """Performs the SGP algorithm.
    Args:
        gn (_type_): Observed star cutout image.
        psf (_type_): PSF matrix.
        bkg (_type_): Background level around the star cutout.
        init_recon (int, optional): Initialization for the reconstructed image.
            Either 0, 1, 2, or 3. Defaults to 0.
        proj_type (int, optional): Type of projection during the iteration.
            Either 0 or 1. Defaults to 0.
        stop_criterion (int, optional): Choice of rule to stop iteration.
            Either 1, 2, 3, or 4. Defaults to 0.
        MAXIT (int, optional): Maximum no. of iterations. Defaults to 500.
        gamma (_type_, optional): Linesearch penalty parameter. Defaults to 1e-4.
        beta (float, optional): Linesearch back-tracking/scaling parameter (used in the backtracking loop). Defaults to 0.4.
        alpha (float, optional): Initial value for alpha, the step length. Defaults to 1.3.
            This value is updated during the iterations.
        alpha_min (_type_, optional): alpha lower bound for Barzilai-Borwein' steplength. Defaults to 1e-5.
        alpha_max (_type_, optional): alpha upper bound for Barzilai-Borwein' steplength. Defaults to 1e5.
        M_alpha (int, optional): Memory length for `alphabb2`. Defaults to 3.
        tau (float, optional): Alternating parameter.. Defaults to 0.5.
        M (int, optional): Non-monotone linear search memory (M = 1 means monotone search). Defaults to 1.
        max_projs (int, optional): Maximum no. of iterations for the flux conservation procedure. Defaults to 1000.
        save (bool, optional): Whether to save the reconstructed image or not. Defaults to True.
        verbose (bool, optional): Controls screen verbosity. Defaults to True.
        flux (_type_, optional): Precomputed flux of the object inside `gn`. Defaults to None.
        scale_data (bool, optional): Whether to scale `gn`, 'bkg`, and `x` (the reconstructed image) before applying SGP. Defaults to False.
        use_original_SGP_Afunction: If True, works only when image and PSF size are same. Set to False if sizes are different.
    Raises:
        OSError: _description_
    Returns:
        _type_: _description_
    Notes:
        == Porting from MATLAB to Numpy ==
        1. np.dot(a, b) is same as dot(a, b) ONLY for 1-D arrays.
        2. np.multiply(a, b) (or a * b) is same as a .* b
        3. If C = [1, 2, 3], then C[0:2] is same as C(1:2).
            In general, array[i:k] in NumPy is same as array(i+1:k) in MATLAB.
        4. x.conj().T in Numpy is the same as x' in matlab, where x is a two-dimensional array.
        * Afunction implementation is provided (similar to SGP-dec). A slightly different approach compared to SGP-dec is also commented out in the code.
        * See here: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for more details.
    """
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e4 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        errmsg = f"\n\tsum(psf) - 1. = {checkPSF}, tolerance = {tolCheckPSF}"
        raise ValueError(f'PSF is not normalized! Provide a normalized PSF! {errmsg}')

    logging.basicConfig(filename='sgp.log', level=logging.INFO, force=True)

    _shape = gn.shape
    if schedule_lr:
        init_lr = lr
    
    if use_original_SGP_Afunction:
        TF = np.fft.fftn(np.fft.fftshift(psf))
        CTF = np.conj(TF)
        def afunction(x, TF, dimensions):
            x = np.reshape(x, dimensions)
            out = np.real(np.fft.ifftn(
                np.multiply(TF, np.fft.fftn(x))
            ))
            out = out.flatten()
            return out

        A = partial(afunction, TF=TF, dimensions=psf.shape)
        AT = partial(afunction, TF=CTF, dimensions=psf.shape)
    else:
        def A(psf, x):
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
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        def AT(psf, x):
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
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf.conj().T, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        A = partial(A, psf=psf)
        AT = partial(AT, psf=psf)

    t0 = timer()  # Start clock timer.

    # Initialization of reconstructed image.
    if init_recon == 0:
        x = np.zeros_like(gn)
    elif init_recon == 1:
        np.random.seed(42)
        x = np.random.randn(*gn.shape)
    elif init_recon == 2:
        x = gn.copy()
    elif init_recon == 3:
        if flux is None:
            x = np.sum(gn - bkg) / gn.size * np.ones_like(gn)
        else:
            x = flux / gn.size * np.ones_like(gn)

    # Treat images as vectors.
    gn = gn.flatten()
    x = x.flatten()
    bkg = bkg.flatten()

    # Stop criterion settings.
    if stop_criterion == 1:
        tol = []
    elif stop_criterion == 2 or stop_criterion == 3:
        tol = tol_convergence
    elif stop_criterion == 4:
        tol = 1 + 1 / np.mean(gn)

    # Scaling
    # !! Note that for stop_criterion=3 where the KL divergence is calculated, it is important to scale the data since KL divergence needs probabilities. In future, it might be helpful to forcefully scale data if stop_criterion=3
    if scale_data:
        scaling = np.max(gn)
        gn = gn / scaling
        bkg = bkg / scaling
        x = x / scaling
    else:
        scaling = 1.   # Scaling can have adverse effects on the flux in the final scaled output image, hence we do not scale.

    # Change null pixels of observed image.
    vmin = np.min(gn[gn > 0])
    eps = np.finfo(float).eps
    gn[gn <= 0] = vmin * eps * eps

    # Computations needed only once.
    N = gn.size
    if flux is None:
        flux = np.sum(gn - bkg)
    else:  # If flux is already provided, we need to scale it. This option is recommended.
        flux /= scaling  # Input a precomputed flux: this could be more accurate in some situations.

    iter_ = 1
    Valpha = alpha_max * np.ones(M_alpha)
    Fold = -1e30 * np.ones(M)
    Discr_coeff = 2 / N * scaling
    # ONE = np.ones(N)

    # Projection type.
    pflag = proj_type  # Default: 0.

    # Setup directory to store reconstructed images.
    if save:
        dirname = "SGP_reconstructed_images/"
        try:
            os.mkdir(dirname)
            fits.writeto(f'{dirname}/orig.fits', gn.reshape(_shape))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError("Directory already exists!")
            pass

    discr = np.zeros(MAXIT + 1)
    times = np.zeros(MAXIT + 1)
    times[0] = 0

    # Start of SGP.
    # Project the initial point.
    if pflag == 0:
        x[x < 0] = 0
    elif pflag == 1:
        # Here an identity matrix is used for projecting the initial point, which means it is not a "scaled" gradient projection as of now.
        # Instead, it is a simple gradient projection without scaling (see equation 2 in https://kd.nsfc.gov.cn/paperDownload/ZD6608905.pdf, for example).
        x = projectDF(flux, x, np.ones_like(x), max_projs=max_projs)

    # Compute objecive function value.
    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = betaDivDerivwrtY(AT, den, gn, betaParam)

    # Divergence.
    # Note: beta div with betaParam=1 and KL divergence equation will match only if flux=None, else slight differences will be there.
    fv = betaDiv(den, gn, betaParam)

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + bkg)), AT(x=gn))
    X_low_bound = np.min(y[y > 0])
    X_upp_bound = np.max(y)
    if X_upp_bound / X_low_bound < 50:
        X_low_bound = X_low_bound / 10
        X_upp_bound = X_upp_bound * 10

    # Discrepancy.
    discr[0] = Discr_coeff * fv

    # Scaling matrix.
    if init_recon == 0:
        X = np.ones_like(x)
    else:
        X = x.copy()
        # Bounds
        X[X < X_low_bound] = X_low_bound
        X[X > X_upp_bound] = X_upp_bound

    if pflag == 1:
        D = np.divide(1, X)

    # Setup tolerance for main SGP iterations.
    if verbose:
        if stop_criterion == 2:
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 0 \n')
            tol = tol * tol
        elif stop_criterion == 3:
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | 0 \n')
        elif stop_criterion == 4:
            logging.info(f'it {iter_-1} D_k {discr[0]} \n')

    # Main loop.
    loop = True
    epoch = 0
    betadivs = []
    while loop:
        epoch += 1
        prev_x = x.copy()

        # Store alpha and obj func values.
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D, max_projs=max_projs)

        d = y - x

        # Backtracking loop for linearsearch.
        gd = np.dot(d, g)
        # In every epoch, lambda is set to 1. If we do not, then lambda will be multiplied many times with beta, which is less than 1, causing lambda to reach zero. This would mean there are no updates on x in later stages.
        lam = 1  # `lam = 1` is a common choice, so we use it.

        fcontinue = 1
        d_tf = A(x=d)
        fr = max(Fold)

        while fcontinue:
            xplus = x + lam * d
            x_tf_try = x_tf + lam*d_tf
            den = x_tf_try + bkg

            # temp = np.divide(gn, den)
            fv = betaDiv(den, gn, betaParam)

            if fv <= fr + gamma * lam * gd or lam < 1e-12:
                x = xplus.copy()
                xplus = None  # clear the variable.
                sk = lam*d
                x_tf = x_tf_try
                x_tf_try = None  # clear
                gtemp = betaDivDerivwrtY(AT, den, gn, betaParam)

                yk = gtemp - g
                g = gtemp.copy()
                gtemp = None  # clear
                fcontinue = 0
            else:
                lam = lam * beta
                if adapt_beta:
                    bgrad = betaDivDeriv(den, gn, betaParam).mean()
                    betaParam = betaParam - lr * bgrad
                    # betaParam = np.mean(np.full(_shape, betaParam).ravel() - lr * bgrad)  # This is another option to update betaParam.

        if fv >= fr and verbose:
            logging.warning("\tWarning, fv >= fr")

        # Update the scaling matrix and steplength
        X = x.copy()
        X[X < X_low_bound] = X_low_bound
        X[X > X_upp_bound] = X_upp_bound

        # assert all(np.isfinite(X)), "The scaling matrix violates either the lower or upper bound!"

        D = np.divide(1, X)
        sk2 = np.multiply(sk, D)
        yk2 = np.multiply(yk, X)
        bk = np.dot(sk2, yk)
        ck = np.dot(yk2, sk)
        if bk <= 0:
            alpha1 = min(10*alpha, alpha_max)
        else:
            alpha1bb = np.sum(np.dot(sk2, sk2)) / bk
            alpha1 = min(alpha_max, max(alpha_min, alpha1bb))
        if ck <= 0:
            alpha2 = min(10*alpha, alpha_max)
        else:
            alpha2bb = ck / np.sum(np.dot(yk2, yk2))
            alpha2 = min(alpha_max, max(alpha_min, alpha2bb))

        Valpha[M_alpha-1] = alpha2

        if iter_ <= 20:
            alpha = min(Valpha)
        elif alpha2/alpha1 < tau:
            alpha = min(Valpha)
            tau = tau * 0.9
        else:
            alpha = alpha1
            tau = tau * 1.1

        # Note: At this point, the original matlab code does: alpha = double(single(alpha)), which we do not do here. Don't know what's the meaning of that.

        if schedule_lr:
            # Update learning rate.
            lr = lr_schedule(init_lr, lr_exp_param, epoch)
            # print(f'Learning rate at epoch {epoch}: {lr}')

        iter_ += 1
        times[iter_-1] = timer() - t0
        discr[iter_-1] = Discr_coeff * fv

        # Stop criterion.
        if stop_criterion == 1:
            logging.info(f'it {iter_-1} of  {MAXIT}\n')
        elif stop_criterion == 2:
            normstep = np.dot(sk, sk) / np.dot(x, x)
            loop = normstep > tol
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 {normstep} tol {tol}\n')
        elif stop_criterion == 3:
            reldecrease = (Fold[M-1]-fv) / fv
            loop = reldecrease > tol and reldecrease >= 0
            betadivs.append(fv)
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | {reldecrease} tol {tol}\n')
        elif stop_criterion == 4:
            loop = discr[iter_-1] > tol
            logging.info(f'it {iter_-1} D_k {discr[iter_-1]} tol {tol}\n')

        if iter_ > MAXIT:
            loop = False

        if save:
            filename = f'SGP_reconstructed_images/rec_{iter_-1}.fits'
            fits.writeto(filename, x.reshape(_shape), overwrite=True)
            # Residual image.
            res = np.divide(x - gn, np.sqrt(x))
            filename = f'SGP_reconstructed_images/res_{iter_-1}.fits'
            fits.writeto(filename, res.reshape(_shape), overwrite=True)

        if not loop:
            x = prev_x

        if epoch == MAXIT:
          break

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    print(f'Beta parameter in beta-divergence (final value): {betaParam}')
    print(f'No. of iterations: {iter_}')

    return x, iter_, discr, times, None