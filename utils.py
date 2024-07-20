import os
import copy
import subprocess
import numpy as np
import pandas as pd

from pandas.errors import EmptyDataError

import glob
from astropy.io import fits
# from astropy.modeling import models, fitting
# from astropy.stats import gaussian_fwhm_to_sigma

from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from photutils.segmentation import SourceFinder

# from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.nddata import Cutout2D

from sklearn.neighbors import BallTree

# from astropy.stats import sigma_clipped_stats, SigmaClip
# from photutils.segmentation import detect_threshold

from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_exact, reproject_interp

# # from photutils.datasets import apply_poisson_noise
# from photutils.utils import calc_total_error
# from photutils.detection import find_peaks

from constants import CAT_COLUMNS

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('paper', font_scale = 2)


def degrade(image, psf):
    # Convolve.
    final = convolve(image, psf, normalize_kernel=True, normalization_zero_tol=1e-4)
    # Add background.
    # final += bkg
    # # Now perturb the image with Poisson noise and readout noise.
    # final = apply_poisson_noise(final, seed=42)
    # Compensate for readout noise.
    # final += readout_noise ** 2
    # bkg += readout_noise ** 2
    return final


# def get_quick_segmap(data, n_pixels=5):
#     """Returns a segmentation map using simple statistics. Not meant to be used for analyses.

#     Args:
#         data (_type_): _description_
#         n_pixels (int, optional): _description_. Defaults to 5.

#     Returns:
#         _type_: _description_
#     """
#     finder = SourceFinder(npixels=n_pixels, progress_bar=False, deblend=True, nproc=1)  # Can pass nproc=None to use all CPUs in a machine.

#     _, median, std = sigma_clipped_stats(data)
#     threshold = 1.5 * std

#     data_bkg_subtracted = data - median  # subtract the background
#     kernel = make_2dgaussian_kernel(1.2, size=5)  # FWHM = 1.2
#     convolved_data = convolve(data_bkg_subtracted, kernel)
#     segment_map = finder(convolved_data, threshold)

#     return segment_map

def radial_profile(data, center):
    """From https://stackoverflow.com/a/34979185. 
    `data` must be background subtracted for more accurate profiles.
    """
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile.tolist()


# def estimate_quick_FWHM(image):
#     bkg = Background2D(image, box_size=50)
#     threshold = detect_threshold(image, nsigma=3.0, background=bkg.background)
#     peaks = find_peaks(image, threshold=threshold, box_size=5)
#     print(peaks)
#     for row in peaks:
#         center = (row['x_peak'], row['y_peak'])
#         cutout = Cutout2D(image, center, size=50).data
#         plt.imshow(cutout)
#         plt.show()
#         radprof = radial_profile(cutout, center)
#         print(radprof)
#         # print(np.where(np.array(radprof) <= radprof[0]/2))


def source_info(
    data, box_width, box_height, min_area=5, threshold=2,
    gain=1.0, plot_positions_indicator=True, use_sextractor=False,
    image_name=None, defaultFile=None, i=0, sextractor_parameters=None,
    original=True, use_subdiv=True
):
    """Returns a catalog for source measurements and properties.

    Args:
        data (numpy.ndarray): 2D Image.
        box_width (int): width of box for background estimation.
        box_height (int): height of box for background estimation.
        bkg (float, numpy.ndarray): Background level.
        min_area (int): Minimum. area for an object to have to be detected as a source.
        threshold (float): threshold value for pixel detection.
        mask (numpy.ndarray): mask for excluding certain pixels (saturated, etc.).

    Returns:
        photutils.segmentation.SourceCatalog: A source catalog object.

    Note
    ----
    - `data` must NOT be background-subtracted.
    - Use smaller box sizes (height and width ~ 64 or so) for stellar fields, but use larger for images with extended sources.

    """
    if use_sextractor:
        os.chdir('sgp_reconstruction_results')
 
        if not use_subdiv:
            prefix = 'orig' if original else 'deconv'
        else:
            prefix = f'subdiv_orig_{i}' if original else f'subdiv_deconv_{i}'

        catalog_name = f'{prefix}_251.fits_scat_sextractor.csv'

        if not original:
            if sextractor_parameters is not None:
                sextractor_parameters_copy = copy.deepcopy(sextractor_parameters)
                sextractor_parameters_copy.update({
                    'BACK_TYPE': 'MANUAL',
                    'BACK_VALUE': 0,  # because deconvolution is estimating f from Af+bkg, so ideally there must not be any background.
                    'DETECT_MINAREA': 1,  # because deconvolved images have no background, so even 1-pixel detection may be non-spurious.
                    'FILTER': 'N',
                    'CLEAN': 'N'
                })

        if sextractor_parameters is None:
            command = ['source-extractor', image_name, '-c', defaultFile, '-CATALOG_NAME', catalog_name, '-CHECKIMAGE_NAME', f'{prefix}_251.fits_scat_sextractor_bkg.fits,{prefix}_251.fits_scat_sextractor_bkgrms.fits']
            print(command)
            subprocess.run(command)
        else:
            command = [
                'source-extractor', image_name, '-c', defaultFile, '-CATALOG_NAME', catalog_name,
                '-CHECKIMAGE_NAME', f'{prefix}_251.fits_scat_sextractor_bkg.fits,{prefix}_251.fits_scat_sextractor_bkgrms.fits',
                '-MAG_ZEROPOINT', f'{sextractor_parameters["MAG_ZEROPOINT"]}',
                '-SEEING_FWHM', f'{sextractor_parameters["SEEING_FWHM"]}', '-GAIN', f'{sextractor_parameters["GAIN"]}',
                '-PIXEL_SCALE', f'{sextractor_parameters["PIXEL_SCALE"]}'
            ]
            if not original:
                command += [
                    '-BACK_TYPE', f'{sextractor_parameters_copy["BACK_TYPE"]}',
                    '-BACK_VALUE', f'{sextractor_parameters_copy["BACK_VALUE"]}',
                    '-DETECT_MINAREA', f'{sextractor_parameters_copy["DETECT_MINAREA"]}',
                    '-FILTER', f'{sextractor_parameters_copy["FILTER"]}',
                    '-CLEAN', f'{sextractor_parameters_copy["CLEAN"]}'
                ]
            print(command)
            subprocess.run(command)

        bkg = fits.getdata(catalog_name.split('.csv')[0] + '_bkg.fits')
        bkgrms = fits.getdata(catalog_name.split('.csv')[0] + '_bkgrms.fits')

        # Sometimes, there can be no source detected in the image. This can happen when
        # the image has no source (is blank, for example) or if the subdivision size used
        # is very small (if using teh subdivision approach) such that no source is present in that subdivision.
        try:
            cat = pd.read_csv(catalog_name, skiprows=len(CAT_COLUMNS), header=None, sep='\s+')#delim_whitespace=True)
        except EmptyDataError:
            cat = None
            os.chdir('../')
            return None, None, bkg, bkgrms, None

        cat.columns = CAT_COLUMNS
        cat.to_csv(catalog_name)  # After adding column names, rewrite to file.
        os.chdir('../')
        return cat, cat['FLUX_ISO'], bkg, bkgrms, None
    else:
        import sep
        data = data.byteswap(inplace=False).newbyteorder()
        bkg = sep.Background(data, bw=box_width, bh=box_height, fw=3, fh=3)
        data_sub = data - bkg
        objects = sep.extract(data_sub, threshold, err=bkg.globalrms, minarea=min_area, gain=gain)  # TODO: Q) Can maskthresh be set to ccd_sat_level?

        fluxes = [objects[i][-8] for i in range(len(objects))]  # -8 index corresponds to flux.
        fig = plot_positions(data_sub, objects) if plot_positions_indicator else None
        return objects, fluxes, bkg.back(), bkg.rms(), fig


def scale_psf(psf, gaussian_fwhm=1.2, size=None):  # For example, seeing_fwhm=2.53319/1.012
    """Returns a 2D Gaussian kernel by scaling the FWHM of psf.

    Args:
        psf (numpy.ndarray): The input PSF.
        gaussian_fwhm (float): FWHM (in pix) of the Gaussian kernel used to convolve the PSF.
        size (tuple): X and Y size of the Gaussian kernel used to convolve the PSF, defaults to None.

    Returns:
        scaled_psf: The scaled version of the PSF.

     Notes:
        * See this talk paper: https://reu.physics.ucsb.edu/sites/default/files/sitefiles/REUPapersTalks/2021-REUPapersTalks/Beck-Dacus-UCSB-REU-paper.pdf

    """
    if size is None:
        size = psf.shape

    kernel = make_2dgaussian_kernel(gaussian_fwhm, size=size)
    scaled_psf = convolve(psf, kernel)

    # Ensure normalization of PSF.
    scaled_psf /= scaled_psf.sum()
    return scaled_psf


def plot_positions(data_sub, objects, use_sextractor=True, figname=None, add_noise=False):
    if use_sextractor:
        # plot background-subtracted image
        fig, ax = plt.subplots(figsize=(10, 8))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # m, s = np.mean(data_sub), np.std(data_sub)
        if add_noise:
            data_sub = data_sub + abs(data_sub[data_sub!=0].min()) * np.abs(np.random.randn(data_sub.shape[0], data_sub.shape[1]))
        im = ax.imshow(data_sub, origin='lower', cmap='viridis')

        if len(objects) > 0:
            # plot an ellipse for each object
            for i in range(len(objects)):
                # NOTE: 1 is subtracted from the detected coordinates since matplotlib follows the (0, 0) convention
                # whereas SExtractor uses the (1, 1) convention.
                e = Ellipse(xy=(objects['XWIN_IMAGE'][i]-1, objects['YWIN_IMAGE'][i]-1),
                            width=6*objects['AWIN_IMAGE'][i],
                            height=6*objects['BWIN_IMAGE'][i],
                            angle=objects['THETAWIN_IMAGE'][i] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
                # ax.text(objects['X_IMAGE'][i], objects['Y_IMAGE'][i], np.round(objects['FLUX_ISO'][i], 2), color='white')

        fig.colorbar(im, cax=cax, orientation='vertical')

        # plt.savefig(figname, bbox_inches='tight')
        plt.show()
        # return fig
    else:
        # plot background-subtracted image
        fig, ax = plt.subplots(figsize=(10, 8))
        m, s = np.mean(data_sub), np.std(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                    vmin=m-s, vmax=m+s, origin='lower')

        # plot an ellipse for each object
        for i in range(len(objects)):
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6*objects['a'][i],
                        height=6*objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        return fig


def validation_source(image, coord, bkgmap, rmsmap, size=100):
    """This validation is mainly designed for source detection on deconvolved images.
    We sometimes observed spurious sources to be detected. This function can help guide.

    Args:
        image (_type_): image, must not be background-subtracted.
        coord (_type_): (x, y) position of the source.
        bkgmap (_type_): 2D background.
        rmsmap (_type_): 2D background RMS.

    """
    source_cutout = Cutout2D(image, coord, size=size, mode='partial', fill_value=0.0).data
    bkg = np.median(Cutout2D(bkgmap, coord, size=size, mode='partial', fill_value=0.0).data)
    rms = np.mean(Cutout2D(rmsmap, coord, size=size, mode='partial', fill_value=0.0).data)
    source_pixs = np.sort(source_cutout.flatten())[-3:].mean()

    return source_pixs > bkg + 3 * rms


def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format

    Credit: https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7

    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def create_subdivisions(image, subdiv_shape=(100, 100), overlap=10, wcs=None):
    # Note: All shapes assume first entry is the height and second entry is width.
    # indices = ndpatch.get_patches_indices(image.shape, subdiv_shape, overlap)
    sliceXYXY = calculate_slice_bboxes(
        image.shape[0], image.shape[1], subdiv_shape[0], subdiv_shape[1],
        overlap/subdiv_shape[0], overlap/subdiv_shape[1]
    )
    subdivs = []
    for s in sliceXYXY:
        cutout = Cutout2D(image, ((s[0]+s[2])/2, (s[1]+s[3])/ 2), size=subdiv_shape, wcs=wcs)
        subdivs.append(cutout)
    return subdivs


def reconstruct_full_image_from_patches(output_projection_header, string_key='image', dirname='sgp_reconstruction_results', original=True):  # string_key can be, for e.g., 'image', 'bkg', or 'bkgrms'.
    prefix = 'orig' if original else 'deconvolved'
    arr, footprint = reproject_and_coadd(
        [fits.open(f)[0] for f in sorted(glob.glob(f'{dirname}/temp_{prefix}_{string_key}*.fits'), key=lambda x: x[17:19])],
        output_projection=output_projection_header, reproject_function=reproject_interp, match_background=False
    )
    return arr, footprint

# def reconstruct_full_image_from_patches_original(output_projection_header, string_key='image', dirname='sgp_reconstruction_results'):  # string_key can be, for e.g., 'image', 'bkg', or 'bkgrms'.
#     arr, footprint = reproject_and_coadd(
#         [fits.open(f)[0] for f in sorted(glob.glob(f'{dirname}/temp_orig_{string_key}*.fits'), key=lambda x: x[17:19])],
#         output_projection=output_projection_header, reproject_function=reproject_interp, match_background=False
#     )
#     return arr, footprint

def add_artificial_sky_background(image, bkg_map):
    """Add sky background to the provided image.

    Parameters
    ----------

    image : numpy array
        Image whose shape the cosmic array should match.
    bkg_map: numpy array
        2D map to add to image as background.

    """
    return image + bkg_map

def run_crossmatching(basename):
    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_for_each_table1_row_1_and_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1and2', 'params=1.383', 'find=best1', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_for_each_table2_row_1_and_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1and2', 'params=1.383', 'find=best2', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_symmetric_1_and_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1and2', 'params=1.383', 'find=best', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_all_matches_1_and_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1and2', 'params=1.383', 'find=all', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_for_each_table1_row_1_not_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1not2', 'params=1.383', 'find=best1', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_for_each_table2_row_1_not_2_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=1not2', 'params=1.383', 'find=best2', 'progress=time'
        ]
    )

    subprocess.run(
        [
            './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
            f'in2=orig_{basename}_scat_sextractor.csv', 'ifmt2=csv',
            f'out={basename}_crossmatched_best_match_for_each_table2_row_2_not_1_stilts.csv',
            'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
            'join=2not1', 'params=1.383', 'find=best2', 'progress=time'
        ]
    )


def remove_very_close_coords(cat, threshold=1., leaf_size=40):
    # About the threshold distance: any duplicate coordinate < `threshold` pixels will be removed.

    coords = cat[['X_IMAGE_DBL', 'Y_IMAGE_DBL']].values
    tree = BallTree(coords, leaf_size=leaf_size)  # default leaf_size=40.

    # Query all points to find neighbors within the threshold
    indices = tree.query_radius(coords, r=threshold)

    # Create a mask to keep track of rows to keep
    mask = np.ones(len(cat), dtype=bool)

    for idx, neighbors in enumerate(indices):
        if mask[idx]:
            # Mark all neighbors as False except the current point
            mask[neighbors[neighbors != idx]] = False

    # Filter the dataframe
    filtered_cat = cat[mask]

    return filtered_cat


import math
def find_closest_factors(N):
    # Start with the integer part of the square root of N
    A = int(math.isqrt(N))
    
    # Decrease A until we find a divisor
    while N % A != 0:
        A -= 1
    
    # Calculate B
    B = N // A
    
    return A, B

import numpy as np
import math

def arrange_2d_arrays(arrays):
    # Determine the number of arrays
    num_arrays = len(arrays)

    # Calculate the size of the grid
    grid_size = math.ceil(math.sqrt(num_arrays))

    # Determine the shape of individual arrays (assuming they all have the same shape)
    array_shape = arrays[0].shape
    rows_per_array, cols_per_array = array_shape

    # Calculate the shape of the big array
    big_array_shape = (grid_size * rows_per_array, grid_size * cols_per_array)
    big_array = np.zeros(big_array_shape)
    big_array += -99

    # Fill the big array with the smaller arrays
    for idx, array in enumerate(arrays):
        row_start = (idx // grid_size) * rows_per_array
        row_end = row_start + rows_per_array
        col_start = (idx % grid_size) * cols_per_array
        col_end = col_start + cols_per_array
        big_array[row_start:row_end, col_start:col_end] = array

    return big_array
