import os
import glob
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

from astropy.wcs import WCS
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import interpolate_replace_nans

import matplotlib.pyplot as plt

from utils import (
    source_info, scale_psf, add_artificial_sky_background, create_subdivisions, reconstruct_full_image_from_patches,
    reconstruct_full_image_from_patches_original, run_crossmatching
)
from constants import CAT_COLUMNS

from sgp import sgp, sgp_betaDiv


def print_options(opt):
    print('\n')
    print("------------ Options ------------")
    for arg in vars(opt):
        print(f'{arg}:\t\t{getattr(opt, arg)}')
    print("------------ End ----------------")
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets data path for data used for SGP')
    parser.add_argument('--data_path_sciimg', type=str, help='Data path that contains the science image. A FITS file.', required=True)
    parser.add_argument('--data_path_psf', type=str, help='Data path that contains the PSF associated with the supplied science image. A FITS file.', required=True)
    # Important note: using scale_psf when not using --degrade_image would mean manipulating the PSF - this will rarely help since it changes the science.
    parser.add_argument('--scale_psf', action='store_true', help='Whether to scale the FWHM of the PSF. A 2D Gaussian PSF is returned and hence the scaled PSF might not have properties exactly the same as input PSF.')
    parser.add_argument('--psf_scale_fwhm', type=float, default=1.2, help='Only used if --scale_psf is specified. It specifies the FWHM of the 2D Gaussian kernel used to generate the scaled PSF.')
    parser.add_argument('--init_recon', type=int, default=2, help='How to initialize the reconstructed image.')
    parser.add_argument('--stop_criterion', type=int, default=3, help='How to decide when to stop SGP iterations.')
    # parser.add_argument('--save_images', action='store_true', help='if specified, store images as FITS files.')
    parser.add_argument('--flip_image', action='store_true', help='if specified, horizontally flips the input image before passing it to SGP.')
    # parser.add_argument('--add_bkg_to_deconvolved', action='store_true', help='if specificed, adds an artificial background to the deconvolved image before detection with the aim to remove any spurious sources.')
    parser.add_argument('--box_height', type=int, default=64, help='height of box for estimating background in the input image given by `data_path_sciimg`, only used if not specified --use_subdiv')
    parser.add_argument('--box_width', type=int, default=64, help='width of box for estimating background in the input image given by `data_path_sciimg`, only used if not specified --use_subdiv')
    parser.add_argument('--use_subdiv', action='store_true', help='If specified, creates subdivisions, deconvolves each of them, and then mosaics them to create a single final deconvolved image.')
    parser.add_argument('--subdiv_size', type=int, default=100, help='subdivision size, only considered if --use_subdiv is specified.')
    parser.add_argument('--subdiv_overlap', type=int, default=0, help='Overlap (in pix) to use while extracting the subdivisions, only considered if --use_subdiv is specified. NOTE: This overlap is used as tried to be used for all subdivisions. When the image height and width are not both an integral multiple of the subdivisions height and width, the rightmost and bottommost subdivisions may have a different overlap than `subdiv_overlap` since in this implementation, we ensure that all subdivisions are of the same size, as provided by `subdivision_size`.')
    parser.add_argument('--sextractor_config_file_name', type=str, help='(Note: This must invariably be in the sgp_reconstruction_results/ directory, but pass only the filename to this argument and not the entire path) Name of the sextractor config file for original image. The config file for the deconvolved images is set based on the original config file. The Only used if use_sextractor is True')
    parser.add_argument('--use_sextractor', action='store_true', help='Whether to use the original SExtractor for extracting source information. *Recommended to set it to True since this script is only reliably tested for that case.*')
    parser.add_argument('--use_beta_div', action='store_true', help='Whether to use beta divergence inside SGP instead of the KL divergence.')
    parser.add_argument('--initial_beta', type=float, default=1.005, help='The initial value of beta to start with.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, help='The initial learning rate to use for updating beta.')
    parser.add_argument('--tol_convergence', type=float, default=1e-4, help='The tolerance level to use for terminating the SGP iterations.')
    parser.add_argument('--gain', type=float, default=None, help='CCD gain')
    parser.add_argument('--saturate', type=float, default=None, help='CCD saturating pixel value.')
    parser.add_argument('--reconstruct_full_image_from_subdivisions', action='store_true', help='If specified, will reconstruct the entire image from subdivisions. This step is a big bottleneck, so use it only when the entire image is required. This does not affect any source properties: everything else is intact.')
    parser.add_argument('--interpolate_bad_pixels', action='store_true', help='If specified, will mask non-finite pixel values (NaN/Inf) or saturated pixels and replace them by interpolating from nearby values.')
    parser.add_argument('--pixel_mask', type=str, default='', help='Data path containing a pixel mask. A FITS file. > 0 corresponds to bad pixels and 0 corresponds to good pixels.')
    parser.add_argument('--perform_catalog_crossmatching', action='store_true', help='If specified, will crossmatch detected sources from the original image and the deconvolved image.')
    # parser.add_argument('--crossmatch_filename_prefix', type=str, help='Prefix to use to save the crossmatched catalogs. Only used if `perform_catalog_crossmatching = True`.')

    opt = parser.parse_args()
    print_options(opt)

    psf_hdul = fits.open(opt.data_path_psf)
    fwhm = psf_hdul[0].header['FWHM']  # in pix.
    psf = psf_hdul[0].data
    if opt.scale_psf:
        # Scale PSF.
        psf = scale_psf(psf, gaussian_fwhm=opt.psf_scale_fwhm, size=psf.shape)

    hdul = fits.open(opt.data_path_sciimg)
    image_header = hdul[0].header
    # readout_noise = image_header['READNOI']  # in e-
    if opt.gain is None:
        gain = image_header['GAIN']  # in e-/ADU
    else:
        gain = opt.gain
    if opt.saturate is None:
        ccd_sat_level = image_header['SATURATE']
    else:
        ccd_sat_level = opt.saturate

    # TODO: Not all images will have the same header keywords. So we can make this optional, only when the user wants, to prevent errors.
    sextractor_parameters = {
        'MAG_ZEROPOINT': image_header['MAGZP'],
        'GAIN': image_header['GAIN'],
        'SEEING_FWHM': image_header['SEEING'] * image_header['PIXSCALE'],
        'PIXEL_SCALE': image_header['PIXSCALE']
    }

    # Get WCS
    wcs = WCS(hdul[0].header)
    image = hdul[0].data
    if opt.flip_image:
        image = np.fliplr(image)
        psf = np.fliplr(psf)

    # Some SGP parameters.
    proj_type = 1

    # Statistics of images.
    dirname = 'sgp_reconstruction_results'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    basename = opt.data_path_sciimg.split('/')[-1]

    if opt.interpolate_bad_pixels:
        if not opt.pixel_mask:  # If a pixel mask is not provided, the below condition is applied.
            pixel_mask = np.logical_or(
                image >= ccd_sat_level, np.logical_not(np.isfinite(image))
            )
            # NOTE: Negative pixels are not handled by this mask. Instead they are set to a very
            # small positive value inside SGP.
        else:  # Otherwise the provided pixel mask is used.
            pixel_mask = fits.getdata(opt.pixel_mask).astype(bool)
        image[pixel_mask] = np.nan
        # Below we use interpolation, but another popular option is to not interpolate,
        # but set masked pixels to zero or to a very small positive value, which is done by `sep` during source extraction,
        # or the lucy routine from STSDAS. See https://articles.adsabs.harvard.edu/pdf/1994ASPC...61..296She
        # The modification in code required would be:
            # ```
            # pixel_mask = np.logical_or(
            #     np.logical_or(
            #         image >= ccd_sat_level, np.logical_not(np.isfinite(image))
            #     ),
            #     image <= 0
            # )
            # vmin = np.min(image[image > 0])
            # eps = np.finfo(float).eps
            # image[pixel_mask] = vmin * eps * eps
            # ```
        image = interpolate_replace_nans(image, psf, convolve=convolve_fft)

    if opt.use_subdiv:
        # NOTE: IMPORTANT: Even if subdiv_overlap != 0, the subdivisions will have overlap
        # if the image size is not an integral multiple of the subdivision size (for both x and y axes).
        # The reason for this is because of the specific way in the subdivisions are created. Our subdivision
        # creation approach prioritizes the shape of the subdivision to match what the user mentions, so if
        # a non-square region is remaining to be extracted, it will still be a square.
        subdivs = create_subdivisions(
            image, subdiv_shape=(opt.subdiv_size, opt.subdiv_size),
            overlap=opt.subdiv_overlap, wcs=wcs
        )

        orig_fluxes = []
        deconv_fluxes = []
        deconv_objects_count = 0
        orig_objects_count = 0
        orig_objects = []
        deconv_objects = []
        execution_times = []

        for i, subdiv in enumerate(subdivs):
            assert subdiv.data.shape == (opt.subdiv_size, opt.subdiv_size)
            if opt.use_sextractor:
                fits.writeto(f'{dirname}/subdiv_{i}_temp.fits', subdiv.data, overwrite=True)

            objects, orig_fluxes_subdiv, orig_bkg, orig_bkg_rms, fig = source_info(
                subdiv.data, opt.subdiv_size, opt.subdiv_size,
                min_area=5, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
                use_sextractor=opt.use_sextractor, image_name=f'subdiv_{i}_temp.fits',
                defaultFile=opt.sextractor_config_file_name, i=i,
                sextractor_parameters=sextractor_parameters, original=True, use_subdiv=True
            )

            if objects is None:
                print(f'\n\nNo source detected in subdivision {i}\n\n')
                # Note that even if no source is detected, we need to still run deconvolution.
                # Since maybe the source existed but was just not detected by SExtractor.
                # The deconvolution may then find one or more new sources if indeed there was a source.
                objects = pd.DataFrame(np.expand_dims(np.ones(len(CAT_COLUMNS)), 0) * -99)
                objects.columns = CAT_COLUMNS
                objects[np.where(np.array(CAT_COLUMNS) == 'FLUX_ISO')[0][0]] = 0.0
                objects[np.where(np.array(CAT_COLUMNS) == 'FLUXERR_ISO')[0][0]] = 0.0
                objects[np.where(np.array(CAT_COLUMNS) == 'BACKGROUND')[0][0]] = 0.0

                # Since there is no source, we cannot sum the fluxes from the SExtractor catalog.
                # Hence, we give a simple flux estimate by sum(pixels - bkg).
                orig_fluxes_subdiv = np.sum(subdiv.data - orig_bkg)

            x_in_nonsubdivided = []
            y_in_nonsubdivided = []
            xpeak_in_nonsubdivided = []
            ypeak_in_nonsubdivided = []
            ra_in_nonsubdivided = []
            dec_in_nonsubdivided = []

            for obj in objects.itertuples():
                # NOTE: 1 is subtracted from the detected coordinates since photutils follows the (0, 0) convention
                # whereas SExtractor follows the (1, 1) convention. See https://github.com/astropy/photutils/blob/main/docs/pixel_conventions.rst
                _x, _y = subdiv.to_original_position((obj.X_IMAGE_DBL-1, obj.Y_IMAGE_DBL-1))
                _xpeak, _ypeak = subdiv.to_original_position((obj.XPEAK_IMAGE-1, obj.YPEAK_IMAGE-1))
                # No need of origin=1 to get 1-based pixel coordinates since 1 is subtracted above already.
                celestial_coords = pixel_to_skycoord(_x, _y, wcs)
                celestial_coords_peak = pixel_to_skycoord(_xpeak, _ypeak, wcs)
                x_in_nonsubdivided.append(_x)
                y_in_nonsubdivided.append(_y)
                xpeak_in_nonsubdivided.append(_xpeak)
                ypeak_in_nonsubdivided.append(_ypeak)
                ra_in_nonsubdivided.append((celestial_coords.ra * u.deg).value)
                dec_in_nonsubdivided.append((celestial_coords.dec * u.deg).value)

            # Note Caveat: XWIN_IMAGE, YWIN_IMAGE, X_IMAGE, and Y_IMAGE are all overwritten using X_IMAGE_DBL and Y_IMAGE_DBL.
            objects['XWIN_IMAGE'] = x_in_nonsubdivided
            objects['YWIN_IMAGE'] = y_in_nonsubdivided
            objects['X_IMAGE'] = x_in_nonsubdivided
            objects['Y_IMAGE'] = y_in_nonsubdivided
            objects['X_IMAGE_DBL'] = x_in_nonsubdivided
            objects['Y_IMAGE_DBL'] = y_in_nonsubdivided
            objects['XWIN_WORLD'] = ra_in_nonsubdivided
            objects['YWIN_WORLD'] = dec_in_nonsubdivided
            objects['X_WORLD'] = ra_in_nonsubdivided
            objects['Y_WORLD'] = dec_in_nonsubdivided
            objects['XPEAK_IMAGE'] = xpeak_in_nonsubdivided
            objects['YPEAK_IMAGE'] = ypeak_in_nonsubdivided

            # Indicator to inform which subdivision were these objects from.
            objects['SUBDIV_NUMBER'] = [i] * len(objects)
            objects['SUBDIV_NUMBER'] = objects['SUBDIV_NUMBER'].astype(int)

            if len(objects) == 1 and objects['NUMBER'].iloc[0] == -99:
                # No source is detected, so `objects` is dummy. So don't add it in detected catalog.
                assert orig_fluxes_subdiv == np.sum(subdiv.data - orig_bkg)
            else:
                orig_objects.append(np.expand_dims(objects, 1))

            if fig is not None:
                fig.savefig(f'{dirname}/orig_{opt.data_path_sciimg.split("/")[-1]}_{i}_positions.png', bbox_inches='tight')
            print(f'No. of objects [subdivision {i}] (original): {len(objects)}')

            # print(np.any(subdiv.data >= ccd_sat_level))
            # print(np.any(np.logical_not(np.isfinite(subdiv.data))))
            # print(np.any(subdiv.data < 0))

            if opt.use_beta_div:
                deconvolved, iterations, _, exec_times, errs = sgp_betaDiv(
                    subdiv.data, psf, orig_bkg, init_recon=opt.init_recon, proj_type=proj_type,
                    stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes_subdiv), scale_data=True,
                    save=False, errflag=False, obj=None, betaParam=opt.initial_beta,
                    lr=opt.initial_lr, lr_exp_param=0.1, schedule_lr=True, tol_convergence=opt.tol_convergence
                )
            else:
                deconvolved, iterations, _, exec_times, errs = sgp(
                    subdiv.data, psf, orig_bkg, init_recon=opt.init_recon, proj_type=proj_type,
                    stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes_subdiv), scale_data=True,  # flux=np.sum(orig_fluxes_subdiv)
                    save=False, errflag=False, obj=None, tol_convergence=opt.tol_convergence
                )

            # if opt.add_bkg_to_deconvolved:
            #     deconvolved_bkg_added = add_artificial_sky_background(deconvolved, orig_bkg)

            deconvolved = deconvolved.byteswap().newbyteorder()
            if opt.use_sextractor:
                fits.writeto(f'{dirname}/subdiv_deconvolved_{i}_temp.fits', deconvolved, overwrite=True)
            deconv_objects_subdiv, deconv_fluxes_subdiv, deconv_bkg, deconv_bkg_rms, fig = source_info(
                deconvolved, deconvolved.shape[1], deconvolved.shape[0],
                min_area=1, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
                use_sextractor=opt.use_sextractor, image_name=f'subdiv_deconvolved_{i}_temp.fits',
                defaultFile=opt.sextractor_config_file_name, i=i,
                sextractor_parameters=sextractor_parameters, original=False, use_subdiv=True
            )

            # from utils import plot_positions
            # plot_positions(subdiv.data, objects, use_sextractor=True, figname=None, add_noise=False)
            # plot_positions(deconvolved, deconv_objects_subdiv, use_sextractor=True, figname=None, add_noise=False)

            if deconv_objects_subdiv is None:
                deconv_objects_subdiv = pd.DataFrame(np.expand_dims(np.ones(len(CAT_COLUMNS)), 0) * -99)
                deconv_objects_subdiv.columns = CAT_COLUMNS
                deconv_objects_subdiv[np.where(np.array(CAT_COLUMNS) == 'FLUX_ISO')[0][0]] = 0.0
                deconv_objects_subdiv[np.where(np.array(CAT_COLUMNS) == 'FLUXERR_ISO')[0][0]] = 0.0
                deconv_objects_subdiv[np.where(np.array(CAT_COLUMNS) == 'BACKGROUND')[0][0]] = 0.0

            x_in_nonsubdivided = []
            y_in_nonsubdivided = []
            xpeak_in_nonsubdivided = []
            ypeak_in_nonsubdivided = []
            ra_in_nonsubdivided = []
            dec_in_nonsubdivided = []

            for obj in deconv_objects_subdiv.itertuples():
                # NOTE: 1 is subtracted from the detected coordinates since photutils follows the (0, 0) convention
                # whereas SExtractor follows the (1, 1) convention. See https://github.com/astropy/photutils/blob/main/docs/pixel_conventions.rst
                _x, _y = subdiv.to_original_position((obj.X_IMAGE_DBL-1, obj.Y_IMAGE_DBL-1))
                _xpeak, _ypeak = subdiv.to_original_position((obj.XPEAK_IMAGE-1, obj.YPEAK_IMAGE-1))
                # No need of origin=1 to get 1-based pixel coordinates since 1 is subtracted above already.
                celestial_coords = pixel_to_skycoord(_x, _y, wcs)
                celestial_coords_peak = pixel_to_skycoord(_xpeak, _ypeak, wcs)
                x_in_nonsubdivided.append(_x)
                y_in_nonsubdivided.append(_y)
                xpeak_in_nonsubdivided.append(_xpeak)
                ypeak_in_nonsubdivided.append(_ypeak)
                ra_in_nonsubdivided.append((celestial_coords.ra * u.deg).value)
                dec_in_nonsubdivided.append((celestial_coords.dec * u.deg).value)

            deconv_objects_subdiv['XWIN_IMAGE'] = x_in_nonsubdivided
            deconv_objects_subdiv['YWIN_IMAGE'] = y_in_nonsubdivided
            deconv_objects_subdiv['X_IMAGE'] = x_in_nonsubdivided
            deconv_objects_subdiv['Y_IMAGE'] = y_in_nonsubdivided
            deconv_objects_subdiv['X_IMAGE_DBL'] = x_in_nonsubdivided
            deconv_objects_subdiv['Y_IMAGE_DBL'] = y_in_nonsubdivided
            deconv_objects_subdiv['XWIN_WORLD'] = ra_in_nonsubdivided
            deconv_objects_subdiv['YWIN_WORLD'] = dec_in_nonsubdivided
            deconv_objects_subdiv['X_WORLD'] = ra_in_nonsubdivided
            deconv_objects_subdiv['Y_WORLD'] = dec_in_nonsubdivided
            deconv_objects_subdiv['XPEAK_IMAGE'] = xpeak_in_nonsubdivided
            deconv_objects_subdiv['YPEAK_IMAGE'] = ypeak_in_nonsubdivided

            # Indicator to inform which subdivision were these objects from.
            deconv_objects_subdiv['SUBDIV_NUMBER'] = [i] * len(deconv_objects_subdiv)
            deconv_objects_subdiv['SUBDIV_NUMBER'] = deconv_objects_subdiv['SUBDIV_NUMBER'].astype(int)

            if len(deconv_objects_subdiv) == 1 and deconv_objects_subdiv['NUMBER'].iloc[0] == -99:
                # No source is detected, so `deconv_objects_subdiv` is dummy. So don't add it in detected catalog.

                # Since there is no source, we cannot sum the fluxes from the SExtractor catalog.
                # Hence, we give a simple flux estimate by sum(pixels - bkg).
                deconv_fluxes.append(np.sum(deconvolved - deconv_bkg))
            else:
                deconv_objects.append(np.expand_dims(deconv_objects_subdiv, 1))
                deconv_fluxes.append(np.sum(deconv_fluxes_subdiv))

            print(f'No. of objects [subdivision {i}] (deconvolved): {len(deconv_objects_subdiv)}')
            deconv_objects_count += len(deconv_objects_subdiv)
            orig_objects_count += len(objects)
            orig_fluxes.append(np.sum(orig_fluxes_subdiv))

            print(f'iterations: {iterations}')

            if i < 10:
                fits.writeto(f'{dirname}/temp_deconvolved_image_0{i}.fits', deconvolved, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkg_0{i}.fits', deconv_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkgrms_0{i}.fits', deconv_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_orig_bkg_0{i}.fits', orig_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_orig_bkgrms_0{i}.fits', orig_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)
            else:
                fits.writeto(f'{dirname}/temp_deconvolved_image_{i}.fits', deconvolved, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkg_{i}.fits', deconv_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkgrms_{i}.fits', deconv_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_orig_bkg_{i}.fits', orig_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_orig_bkgrms_{i}.fits', orig_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)

            execution_times.append(exec_times[-1])

            # NOTE: Uncomment the below lines if you want to debug.
            # fig,ax=plt.subplots(1,2)
            # from astropy.visualization import MinMaxInterval, ImageNormalize, SqrtStretch, ZScaleInterval, PercentileInterval
            # norm1 = ImageNormalize(data=subdiv.data, interval=PercentileInterval(98.), stretch=SqrtStretch())
            # norm2 = ImageNormalize(data=deconvolved, interval=PercentileInterval(99.), stretch=SqrtStretch())
            # ax[0].imshow(subdiv.data, norm=norm1)
            # ax[1].imshow(deconvolved, norm=norm2)
            # plt.show()

        if opt.reconstruct_full_image_from_subdivisions:
            # Reconstruct the subdivisions into a single image.
            # TODO: replace with a less accurate and much faster mosaicing here for experimentation. For the final run, use the most accurate.
            # NOTE: The reconstruct function may fail (i.e., give OSError: Too many open files) for small-sized subdivisions
            # (e.g., 100x100 for a 3kx3k image, giving ~1000 FITS files, one for each subdivision)
            # since in this case, the no. of subdivisions and hence the no. of FITS files becomes large.
            # It is recommended to not use `reconstruct_full_image_from_subdivisions` for small-sized subdivisions since reconstructing full
            # image from subdivisions is used for visualization only and does not affect source statistics and crossmatches.
            t0_recon = timer()
            deconvolved, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="image")
            deconvolved_bkg, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="bkg")
            deconvolved_bkg_rms, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="bkgrms")

            orig_bkg, _ = reconstruct_full_image_from_patches_original(hdul[0].header, string_key="bkg")
            orig_bkg_rms, _ = reconstruct_full_image_from_patches_original(hdul[0].header, string_key="bkgrms")

            assert deconvolved.shape == image.shape
            assert deconvolved_bkg.shape == image.shape
            assert deconvolved_bkg_rms.shape == image.shape
            assert orig_bkg.shape == image.shape
            assert orig_bkg_rms.shape == image.shape

            t_recon = timer() - t0_recon
            print(f'Execution time [all subdivisions] + mosaicking: {np.sum(execution_times) + t_recon} seconds.')

        print(f'Execution time [all subdivisions]: {np.sum(execution_times)} seconds.')

        # Stack all the subdivision objects into a single array.
        deconv_objects = np.squeeze(np.vstack(deconv_objects))
        orig_objects = np.squeeze(np.vstack(orig_objects))
    else:
        fits.writeto(os.path.join(dirname, f'orig_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))

        orig_objects, orig_fluxes, orig_bkg, orig_bkg_rms, fig = source_info(
            image, opt.box_width, opt.box_height, min_area=5, threshold=3, gain=gain, plot_positions_indicator=False,  #  maskthresh=ccd_sat_level
            use_sextractor=opt.use_sextractor, image_name=f'orig_{basename}', defaultFile=opt.sextractor_config_file_name,
            sextractor_parameters=sextractor_parameters, original=True, use_subdiv=False
        )

        if fig is not None:
            fig.savefig(f'{dirname}/orig_{opt.data_path_sciimg.split("/")[-1]}_positions.png', bbox_inches='tight')
        print(f'No. of objects (original): {len(orig_objects)}')

        if opt.use_beta_div:
            deconvolved, iterations, _, exec_times, errs = sgp_betaDiv(
                image, psf, orig_bkg, init_recon=opt.init_recon, proj_type=proj_type,
                stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes), scale_data=True,
                save=False, errflag=False, obj=None, betaParam=opt.initial_beta,
                lr=opt.initial_lr, lr_exp_param=0.1, schedule_lr=True, tol_convergence=opt.tol_convergence
            )
        else:
            deconvolved, iterations, _, exec_times, errs = sgp(
                image, psf, orig_bkg, init_recon=opt.init_recon, proj_type=proj_type,
                stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes), scale_data=True,
                save=False, errflag=False, obj=None, tol_convergence=opt.tol_convergence
            )
        print(f'Execution time: {exec_times[-1]:.2f} seconds.')

    # if opt.add_bkg_to_deconvolved and not opt.use_subdiv and opt.reconstruct_full_image_from_subdivisions:  # When subdivision approach is used, background is already added during each subdivision deconvolution.
    #     deconvolved = add_artificial_sky_background(deconvolved, orig_bkg)

    ## The below two lines are only very needed for very particular cases. But here we keep it general for all cases.
    # import sep
    # sep.set_sub_object_limit(3000)
    # mask = np.ma.array(deconvolved, mask=deconvolved > ccd_sat_level).mask
    if not opt.use_subdiv:
        deconvolved = deconvolved.byteswap().newbyteorder()
        # First write deconvolved image, then update its header, then again write deconvolved image after header update.
        fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True)
        deconv_header = fits.open(os.path.join(dirname, f'deconvolved_{basename}'))[0].header
        for item in wcs.to_header().items():
            deconv_header.append(item)
        fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True, header=deconv_header)
        deconv_objects, deconv_fluxes, deconvolved_bkg, deconvolved_bkg_rms, fig = source_info(
            deconvolved, opt.box_width, opt.box_height, min_area=1, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
            use_sextractor=opt.use_sextractor, image_name=f'deconvolved_{basename}',
            defaultFile=None if not opt.use_sextractor else opt.sextractor_config_file_name,
            sextractor_parameters=sextractor_parameters, original=False, use_subdiv=False
        )
        if fig is not None:
            fig.savefig(f'{dirname}/deconvolved_{opt.data_path_sciimg.split("/")[-1]}_positions.png', bbox_inches='tight')
        print(f'No. of objects (deconvolved): {len(deconv_objects)}')
        fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
        # TODO: Make sure below line no err is raised.
        fits.writeto(os.path.join(dirname, f'deconv_bkgrms_{basename}'), deconvolved_bkg_rms, overwrite=True)

        fits.writeto(os.path.join(dirname, f'orig_bkg_{basename}'), orig_bkg, overwrite=True)
        fits.writeto(os.path.join(dirname, f'orig_bkgrms_{basename}'), orig_bkg_rms, overwrite=True)

    if opt.use_sextractor and opt.use_subdiv:
        CAT_COLUMNS.append('SUBDIV_NUMBER')  # Because we added `SUBDIV_NUMBER` when using subdivisions.
        # pd.DataFrame(data=orig_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
        # pd.DataFrame(data=deconv_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')

        _o = pd.DataFrame(data=orig_objects, columns=CAT_COLUMNS)
        _o.columns = CAT_COLUMNS
        _o.to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
        _d = pd.DataFrame(data=deconv_objects, columns=CAT_COLUMNS)
        _d.columns = CAT_COLUMNS
        _d.to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')

        # Once the final catalog is created, remove all subdivision files to clean up space.
        for img in glob.glob(os.path.join(dirname, 'subdiv*.fits')):
            os.remove(img)
        for img in glob.glob(os.path.join(dirname, 'subdiv*.csv')):
            os.remove(img)

        ############################ COMMENTING BELOW LINES ##############################
        # If there is some overlap or if the entire image size is not an integral multiple of the subdivision size, same sources can be detected across two adjacent subdivisions. Here, we remove duplicate rows.
        # Note that, in the current implementation, each subdivision will be a square irrespective of whether the image size is an integral multiple of the subdivision size or not.
        # This means that in such a case, the subdivisions will have an overlap even if subdiv_overlap=0.
        # Practically, it's very common that the image size is not an integral multiple of the subdivision size.
        _orig_source_cat = pd.read_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
        _deconv_source_cat = pd.read_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')
        _num_orig_sources = _orig_source_cat.shape[0]
        _num_deconv_sources = _deconv_source_cat.shape[0]

        # Set object IDs manually since the IDs were based on subdivisions only. To make them unique, we need to do the below preprocessing.
        def set_id(old_id, subdiv_number):
            return f'{int(subdiv_number)}_{int(old_id)}'
        _orig_source_cat['NUMBER'] = [set_id(_orig_source_cat.loc[idx, 'NUMBER'], _orig_source_cat.loc[idx, 'SUBDIV_NUMBER']) for idx in range(len(_orig_source_cat))]
        _orig_source_cat['ID_PARENT'] = [set_id(_orig_source_cat.loc[idx, 'ID_PARENT'], _orig_source_cat.loc[idx, 'SUBDIV_NUMBER']) for idx in range(len(_orig_source_cat))]
        _deconv_source_cat['NUMBER'] = [set_id(_deconv_source_cat.loc[idx, 'NUMBER'], _deconv_source_cat.loc[idx, 'SUBDIV_NUMBER']) for idx in range(len(_deconv_source_cat))]
        _deconv_source_cat['ID_PARENT'] = [set_id(_deconv_source_cat.loc[idx, 'ID_PARENT'], _deconv_source_cat.loc[idx, 'SUBDIV_NUMBER']) for idx in range(len(_deconv_source_cat))]

        # Remove redundant rows, i.e. rows with the same (x, y) location of the source.
        # Note that we are using the X and Y coordinates instead of ID_PARENT/NUMBER because the aim of removing these duplicates is that since we have subdivision overlap, coinciding sources
        # from adjacent subdivisions will be detected twice. We can only use coordinates to remove them since because they are in different subdivisions, ID_PARENT/NUMBER will never be the same, so it cannot be used for removal.
        # 1. FIRST FILTERING (using sub-pixel-precision coordinates).
        _orig_source_cat.drop_duplicates(subset=['X_IMAGE_DBL', 'Y_IMAGE_DBL'], inplace=True, keep='first')  # Only the first occurrence is retained.
        _deconv_source_cat.drop_duplicates(subset=['X_IMAGE_DBL', 'Y_IMAGE_DBL'], inplace=True, keep='first')  # Only the first occurence is retained.
        # 2. SECOND FILTERING (using integer coordinates).
        # This second filtering is used because two sources from different subdivisions may correspond to the same actual source (even if subdiv_overlap=0)
        # So, this condition will remove such duplicates. The first condition above removes exact duplicates, but this condition tries to remove near-duplicates.
        # Note that we don't check the flux/mag of the sources before removing duplicates: ideally, we only want to remove those whose XPEAK and YPEAK match but also their magnitudes
        # to increase the chances of knowing it's the same source. Doing that is more trustworthy, but what we do below may also be sufficient and okay.
        _orig_source_cat.drop_duplicates(subset=['XPEAK_IMAGE', 'YPEAK_IMAGE'], inplace=True, keep='first')  # Only the first occurrence is retained.
        _deconv_source_cat.drop_duplicates(subset=['XPEAK_IMAGE', 'YPEAK_IMAGE'], inplace=True, keep='first')  # Only the first occurence is retained.
        _num_orig_sources_after = _orig_source_cat.shape[0]
        _num_deconv_sources_after = _deconv_source_cat.shape[0]

        print(f'[Original]: {_num_orig_sources_after} sources out of {_num_orig_sources} remaining after removing duplicates')
        print(f'[Deconvolved]: {_num_deconv_sources_after} sources out of {_num_deconv_sources} remaining after removing duplicates')

        # Note that while the above handles cases where the coordinates in two rows exactly match, sometimes the coordinates are not exact but instead equal up to an error tolerance.
        # For matching for each row in Table 2, for example, ideally we want these two subsets of the same shape: df['X_IMAGE_DBL_2'].unique().shape, df['X_IMAGE_DBL_2'].shape. Same for y-coordinates.
        # But if these two are not of the same shape, then in this code this means that the coordinates were almost equal (within some tolerance) but not exactly the same, and that other source properties match.
        # However, such cases are delicate since it might happen these sources are very nearby (blended).
        # So NOTE: in this code, we refrain from manipulating such cases which means (for crossmatching for each row in Table 2), df['X_IMAGE_DBL_2'].unique().shape, df['X_IMAGE_DBL_2'].shape may not be the same.
        # This means the same original source will kind of repeated. But the no. of such cases is not too high since we have atleast removed the exact duplicate rows above.
        # Now write to file.

        # Remove rows in the original source catalog that have area < 5.
        _orig_source_cat = _orig_source_cat.drop(_orig_source_cat[_orig_source_cat.ISOAREAF_IMAGE < 5].index)
        print(f'{_orig_source_cat.shape[0]} rows remaining out of {_num_orig_sources_after} rows in the original catalog after filtering out rows with ISOAREAF < 5')

        _orig_source_cat.to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
        _deconv_source_cat.to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')
        ##################################################################################
    else:
        columns = [
            'thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy',
            'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', 'cxx', 'cyy', 'cxy', 'cflux', 'flux',
            'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag'
        ]
        pd.DataFrame(data=orig_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat.csv')
        pd.DataFrame(data=deconv_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat.csv')

    print(f'Total flux (before): {np.sum(orig_fluxes)}')
    print(f'Total flux (after): {np.sum(deconv_fluxes)}')

    # Save images and results.
    if opt.use_subdiv:
        fits.writeto(os.path.join(dirname, f'orig_subdiv_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))

        if opt.reconstruct_full_image_from_subdivisions:
            # First write deconvolved image, then update its header, then again write deconvolved image after header update.
            fits.writeto(os.path.join(dirname, f'deconvolved_subdiv_{basename}'), deconvolved, overwrite=True)
            deconv_header = fits.open(os.path.join(dirname, f'deconvolved_subdiv_{basename}'))[0].header
            for item in wcs.to_header().items():
                deconv_header.append(item)
            fits.writeto(os.path.join(dirname, f'deconvolved_subdiv_{basename}'), deconvolved, overwrite=True, header=deconv_header)

            fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
            fits.writeto(os.path.join(dirname, f'deconv_bkgrms_{basename}'), deconvolved_bkg_rms, overwrite=True)

            # NOTE: (if using the subdivision approach): Although the saved original image is as it is, the background and its RMS images are not
            # the ones estimated using the entire image, but rather it's a mosaic of bkg and RMS estimated from each subdivision.
            fits.writeto(os.path.join(dirname, f'orig_bkg_{basename}'), orig_bkg, overwrite=True)
            fits.writeto(os.path.join(dirname, f'orig_bkgrms_{basename}'), orig_bkg_rms, overwrite=True)

        # Note: If we use the subdivision approach, then the below type of background files are not needed anymore.
        for img in glob.glob('*.fits_scat_sextractor_bkg.fits'):
            os.remove(img)

    # else:
    #     fits.writeto(os.path.join(dirname, f'orig_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))
    #     fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True)
    #     fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
    #     fits.writeto(os.path.join(dirname, f'deconv_rms_{basename}'), deconvolved_bkg.rms(), overwrite=True)

    # fits.writeto(os.path.join(dirname, f'orig_bkg_{basename}'), orig_bkg, overwrite=True)
    # fits.writeto(os.path.join(dirname, f'orig_rms_{basename}'), orig_bkg_rms, overwrite=True)

    # Remove temporary orig and deconvolved images.
    for img in glob.glob(f'{dirname}/temp_deconvolved_*.fits'):
        os.remove(img)
    for img in glob.glob(f'{dirname}/temp_orig_*.fits'):
        os.remove(img)
    if opt.use_sextractor:
        for img in glob.glob(f'{dirname}/subdiv_*_temp.fits'):
            os.remove(img)

    exec_times_file = f"{dirname}/execution_times.txt"
    if os.path.exists(exec_times_file) and os.stat(exec_times_file).st_size == 0:
        with open(exec_times_file, "w") as f:
            f.write(f'{opt.data_path_sciimg},{exec_times[-1]},{image.shape[1]},{image.shape[0]},{len(orig_objects)}\n')
    else:
        with open(exec_times_file, "a") as f:
            f.write(f'{opt.data_path_sciimg},{exec_times[-1]},{image.shape[1]},{image.shape[0]},{len(orig_objects)}\n')

    if opt.reconstruct_full_image_from_subdivisions:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from astropy.visualization import ImageNormalize, LogStretch, ZScaleInterval

        norm = ImageNormalize(stretch=LogStretch(), interval=ZScaleInterval())
        fig, ax = plt.subplots(1, 2, figsize=(20, 15))
        im0 = ax[0].imshow(image, origin='lower', norm=norm)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')
        ax[0].set_title('(a) Original image (from ZTF)', fontsize=12)

        im2 = ax[1].imshow(deconvolved, origin='lower', norm=norm)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax[1].set_title('Result of deconvolution', fontsize=12)

        plt.show()

    if opt.perform_catalog_crossmatching:
        # The below crossmatching function (defined in `utils`) is ad-hoc.
        # It runs seven types of crossmatchings on the original and deconvolved source catalogs.
        # It requires Topcat to be installed. Since Topcat is only GUI, this function uses STILTS,
        # which is a command-line
        cwd = os.getcwd()
        os.chdir(dirname)
        run_crossmatching(basename=basename)
        os.chdir(cwd)
