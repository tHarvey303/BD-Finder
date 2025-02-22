# README for fit_sonora.py
# 2025.02.22
# Thomas Harvey

This script takes a catalog and fits the Sonora Bobcat, Cholla, Diamondback and Elf Owl models, and the LOW-Z models.
Sonora Bobcat and Cholla models are provided by Marley et al. (2021, Astrophysical Journal, in press.) and Karalidi et al. (2021, Astrophysical Journal, in press.). They are automatically downloaded from Zenodo if not found in directory code is run in.



Code should be unpacked into a directory, which is where output and data files will be stored. By default sonora data is stored in 'sonora_data/'. base_path in convolve_sed_v2 should be changed to location of jwst_filters file if not in same location as code. 

On first run the code will take a while, as it downloads and unpacks data files (takes approx 20 Gb when unpacked). On subsequent runs the spectra are resampled for speed, so they will be much quicker. 

Changes to code may need to be made to reflect catalogues column names for photometry. Style of EPOCHS catalogues for flux 'FLUX_APER_{band}_aper_corr_Jy', reflecting that the value is aperture corrected and in Jy. For flux error the form is f'FLUXERR_APER_{band}_loc_depth_10pc_Jy', reflecting that the flux uncertainity is derived from a local estimate of the depth, with a 10pc error floor, in Jy. In addition both of these columns have the shape (N, 5), as we measure aperture photometry in 5 circular apertures (radius 0.16, 0.25, 0.5, 0.75, 1 arcsec.) Changes to lines 220, 357, 359 may be required to fit a catalogue with a different format. 

Code is still a work in progress. Contact the author at: thomas.harvey-3@manchester.ac.uk for help, improvements, problems, bugs, complaints etc...

