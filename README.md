# Deconvolution for Large Astronomical Surveys using the Scaled Gradient Projection method
> The code is tested majorly when `opt.use_sextractor` and `opt.use_subdiv` are used to run the code.

This repository contains code for image deconvolution using the Scaled Gradient Projection method (SGP) and has been designed for applications in astronomy. The code currently can perform single-image deconvolution with a known Point Spread Function.

## Motivation
Ground-based astronomical observations are degraded by several factors such as atmospheric seeing, instrumental aberrations, diffraction, and other sources of noise. Deconvolution can help reverse these effects and extract more science from astronomical images than we currently can. 

## Scientific details

Coming soon...

## Usage

- `run.py` contains the driver code.
- `sgp.py` contains the SGP algorithm (see the function `sgp`). There is also a function named `sgp_betaDiv`. It is
the SGP algorithm with beta divergence (see [this paper](https://www.sciencedirect.com/science/article/abs/pii/S2213133723000549)), but is not used in this work.
- `Afunction.py` describes procedures for convolution with PSF.
- `flux_conserve_proj.py` contains the implementation of the projection step in SGP.
- `utils.py` contains some utility functions and `constants.py` defines some constants.
- `richardson_lucy.py` contains the Richardson-Lucy algorithm implementation.
- `ZTF_Deconvolution_Run.ipynb` demonstrates an example to use the SGP algorithm for deconvolving ZTF images.
- `ZTF_Deconvolution_Analysis.ipynb` illustrates the analysis of deconvolution performance (analysis of crossmatched catalogs, visualizations, etc.) and also code for generating plots in the paper.
- The folder `sgp_reconstruction_results` contains SExtractor parameter and configuration files. This is also the folder where the outputs will be stored after deconvolution is run.

## License
[MIT](https://github.com/Yash-10/deconv_ztf/blob/main/LICENSE)
