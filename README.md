# Deconvolution for Large Astronomical Surveys using the Scaled Gradient Projection method
> The code is tested majorly when `opt.use_sextractor` and `opt.use_subdiv` are used to run the code.

Ground-based astronomical observations are degraded by several factors such as atmospheric seeing, instrumental aberrations, diffraction, and other sources of noise. Deconvolution can help reverse these effects and extract more science from astronomical images than we currently can. 

This repository contains code for image deconvolution using the Scaled Gradient Projection method (SGP) and has been designed for applications in astronomy. The code currently can perform single-image deconvolution with a known Point Spread Function.

## Abstract

The arXiv link to the paper is here: TODO

Ground-based astronomical observations will continue to produce resolution-limited images due to atmospheric seeing. Deconvolution reverses such effects and thus can benefit extracted science in multifaceted ways. We apply the Scaled Gradient Projection (SGP) algorithm for the single-band deconvolution of several observed images from the Zwicky Transient Facility and mainly discuss the performance on stellar sources. The method shows good photometric flux preservation, which deteriorates for fainter sources but significantly reduces flux uncertainties even for the faintest sources. Deconvolved sources have a well-defined Full-Width-at-Half-Maximum (FWHM) of roughly one pixel (one arcsecond for ZTF) regardless of the observed seeing. Detection after deconvolution results in catalogs with ≳99.6% completeness relative to detections in the observed images. A few observed sources that could not be detected in the deconvolved image are found near saturated sources, whereas for others, the deconvolved counterparts are detected when slightly different detection parameters are used. The deconvolution reveals new faint sources previously undetectable, which are confirmed by crossmatching with the deeper DESI Legacy DR10 and with Pan-STARRS1 through forced photometry. The method could identify examples of serendipitous potential deblends that exceeded SExtractor's deblending capabilities, with as extreme as ∆m ≈ 3 and separations as small as one arcsecond between the deblended components. Our survey-agnostic approach is better and eight times faster than Richardson-Lucy deconvolution and could be a reliable method for incorporation into survey pipelines.

## Code description

**Scripts**:
- `run.py` contains the driver code.
- `sgp.py` contains the SGP algorithm (see the function `sgp`). There is also a function named `sgp_betaDiv`. It is
the SGP algorithm with beta divergence (see [this paper](https://www.sciencedirect.com/science/article/abs/pii/S2213133723000549)), but is not used in this work.
- `Afunction.py` describes procedures for convolution with PSF.
- `flux_conserve_proj.py` contains the implementation of the projection step in SGP.
- `utils.py` contains some utility functions and `constants.py` defines some constants.
- `richardson_lucy.py` contains the Richardson-Lucy (RL) algorithm implementation.

**Notebooks**
- `ZTF_Deconvolution_Run.ipynb` demonstrates an example to use the SGP algorithm for deconvolving ZTF images.
- `ZTF_Deconvolution_Analysis.ipynb` illustrates the analysis of deconvolution performance (analysis of crossmatched catalogs, visualizations, etc.) and also code for generating plots in the paper.

**Folders**
- `sgp_reconstruction_results` contains SExtractor parameter and configuration files. This is also the folder where the outputs will be stored after deconvolution is run.
- `sgp_results` contains catalogs of observed and deconvolved images considered in this study, and also the crossmatched catalogs.
- `rl_spatialreg` contains RL run log and source catalogs of observed, deconvolved, and crossmatched catalogs for a single field (with ID 626).
- `deconv_validation` contains the analysis notebook for validating deconvolved sources with DESI sources.
    - `deconv_validation/final_catalogs_for_plots` contains catalogs used for making plots for the paper. Each of these is created within the notebook called `deconv_validation/deconv_source_validation.ipynb`.
    - `deconv_validation/forced_phot_lcs` contains ZTF forced photometry light curves, which are used in the above notebook.

## Caveats
- When the subdivision approach is used, `XWIN_IMAGE`, `YWIN_IMAGE`, `X_IMAGE`, and `Y_IMAGE` are all overwritten by `X_IMAGE_DBL` and `Y_IMAGE_DBL`.

## Citation
If you use this code in your work, please cite our paper:

TODO


## License
[MIT](https://github.com/Yash-10/deconv_ztf/blob/main/LICENSE)
