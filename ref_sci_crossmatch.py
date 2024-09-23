# TODO: Ensure code here is same as in repo. git pull actually.
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

from utils import create_subdivisions, source_info
from constants import CAT_COLUMNS

prefix = 'ztf_000626_zr_c12_q2_refimg'
data_path_refimg = f'/kaggle/working/{prefix.split("/")[-1]}sciimg.fits'
ref = fits.open(data_path_refimg)[0]
# sci = fits.open('')[0]
ref_header = ref.header

basename = data_path_refimg.split('/')[-1]

# Options
subdiv_size = 512
use_sextractor = True
use_subdiv = True
subdiv_overlap = 10
sextractor_config_file_name = 'sextractor_config.fits_scat_sextractor.sex'
sextractor_parameters = {
    'MAG_ZEROPOINT': ref_header['MAGZP'],
    'GAIN': ref_header['GAIN'],
    'SEEING_FWHM': ref_header['SEEING'] * ref_header['PIXSCALE'],
    'PIXEL_SCALE': ref_header['PIXSCALE']
}

wcs = WCS(ref[0].header)
subdivs = create_subdivisions(
    ref.data, subdiv_shape=(subdiv_size, subdiv_size),
    overlap=subdiv_overlap, wcs=wcs
)

dirname = 'sgp_reconstruction_results'
dtype = np.float32
orig_objects = []

for i, subdiv in enumerate(subdivs):
    assert subdiv.data.shape == (subdiv_size, subdiv_size)
    if use_sextractor:
        fits.writeto(f'{dirname}/subdiv_{i}_temp.fits', subdiv.data.astype(dtype), overwrite=True)

    objects, orig_fluxes_subdiv, orig_bkg, orig_bkg_rms, fig = source_info(
        ref.data, subdiv_size, subdiv_size,
        use_sextractor=use_sextractor, image_name=f'subdiv_{i}_temp.fits',
        defaultFile=sextractor_config_file_name, i=i,
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
        orig_fluxes_subdiv = np.sum(ref.data - orig_bkg)
    
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

orig_objects = np.squeeze(np.vstack(orig_objects))

if use_sextractor and use_subdiv:
    CAT_COLUMNS.append('SUBDIV_NUMBER')  # Because we added `SUBDIV_NUMBER` when using subdivisions.
    # pd.DataFrame(data=orig_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
    # pd.DataFrame(data=deconv_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')

    _o = pd.DataFrame(data=orig_objects, columns=CAT_COLUMNS)
    _o.columns = CAT_COLUMNS
    _o.to_csv(f'{os.path.join(dirname, f"ref_{basename}")}_scat_sextractor.csv')


import subprocess
subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_for_each_table1_row_1_and_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1and2', 'params=1.383', 'find=best1', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_for_each_table2_row_1_and_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1and2', 'params=1.383', 'find=best2', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_symmetric_1_and_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1and2', 'params=1.383', 'find=best', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_all_matches_1_and_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1and2', 'params=1.383', 'find=all', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_for_each_table1_row_1_not_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1not2', 'params=1.383', 'find=best1', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_for_each_table2_row_1_not_2_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=1not2', 'params=1.383', 'find=best2', 'progress=time'
    ]
)

subprocess.run(
    [
        './topcat', '-stilts', 'tmatch2', f'in1=deconv_{basename}_scat_sextractor.csv', 'ifmt1=csv',
        f'in2=ref_{basename}_scat_sextractor.csv', 'ifmt2=csv',
        f'out={basename}_crossmatched_best_match_for_each_table2_row_2_not_1_stilts_ref.csv',
        'matcher=2d', 'values1=X_IMAGE_DBL Y_IMAGE_DBL', 'values2=X_IMAGE_DBL Y_IMAGE_DBL',
        'join=2not1', 'params=1.383', 'find=best2', 'progress=time'
    ]
)
