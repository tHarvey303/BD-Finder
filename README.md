# BD Fitter


This python package is for fitting brown dwarf (substellar mass object) photometry.  It currently supports the Sonora Bobcat, Cholla, Diamondback and Elf Owl models, and the LOW-Z models.
Photometry for a range of filters is provided, and if more are needed the models are automatically downloaded from Zenodo/Harvard DataVerse if not found in directory code is run in.

Model links: 

[Sonora Bobcat](https://zenodo.org/records/5063476) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac141d))

[Sonora Cholla](https://zenodo.org/records/4450269) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac3140/meta))

[Sonora Diamondback](https://zenodo.org/records/12735103) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad71d5/meta))

Sonora Elf Owl ([Y-Type](https://zenodo.org/records/10381250), [T-Type](https://zenodo.org/records/10385821), [L-Type](https://zenodo.org/records/10385987), [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad18c2/meta))

[LOWZ](https://doi.org/10.7910/DVN/SJRXUO) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac013c))

Please cite all fitted model papers and repositories DOIs if you use this code. 

## Installation

Clone this repository into a folder, and then do pip install . to install. 

An example of use is available in examples/chi2_testing.ipynb. All the user needs to provide is an array of flux and flux uncertainity, and a list of bands (in SVO filter format, or abbreviated will work if HST/NIRcam filters).

Code is still a work in progress. Contact the author at: thomas.harvey-3@manchester.ac.uk for help, improvements, problems, bugs, complaints etc...

