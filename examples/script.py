import astropy.units as u
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from BDFit import StarFit
from astropy.table import Table
import fnmatch

def find_bands(table, flux_wildcard='FLUX_APER_*_aper_corr'):#, error_wildcard='FLUXERR_APER_*_loc_depth'):
    # glob-like matching for column names
    flux_columns = fnmatch.filter(table.colnames, flux_wildcard)
    # get the band names from the column names
    flux_split = flux_wildcard.split('*')
    flux_bands = [col.replace(flux_split[0], '').replace(flux_split[1], '') for col in flux_columns]
    return flux_bands

def provide_phot(table, bands=None, flux_wildcard='FLUX_APER_*_aper_corr_Jy', error_wildcard='FLUXERR_APER_*_loc_depth_10pc_Jy', min_percentage_error=0.1, flux_unit=u.Jy, multi_item_columns_slice=None):
    
    if bands is None:
        bands = find_bands(table)

    flux_columns = [flux_wildcard.replace('*', band) for band in bands]
    error_columns = [error_wildcard.replace('*', band) for band in bands]

    assert all([col in table.colnames for col in flux_columns]), f'Flux columns {flux_columns} not found in table'
    assert all([col in table.colnames for col in error_columns]), f'Error columns {error_columns} not found in table'

    if multi_item_columns_slice is not None:
        raise NotImplementedError('Do this I guess.')

    fluxes = structured_to_unstructured(table[flux_columns].as_array()) * flux_unit
    errors = structured_to_unstructured(table[error_columns].as_array()) * flux_unit

    mask = ((errors / fluxes) < min_percentage_error) & (fluxes > 0)
    errors[mask] = fluxes[mask] * min_percentage_error

    return fluxes, errors

if __name__ == "__main__":

    s = StarFit(verbose=True)

    catalog = '/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/ACS_WFC+NIRCam/JOF/(0.32)as/JOF_MASTER_Sel-F277W+F356W+F444W_v11.fits'
    table = Table.read(catalog)
    bands = find_bands(table, flux_wildcard='FLUX_APER_*_aper_corr_Jy')

    s.fit_catalog(provide_phot, photometry_function_kwargs={'table': table}, bands=bands, dump_fit_results=True, dump_fit_results_path='fit_results.pkl');

    fig, ax = s.plot_fit(201, flux_unit=u.nJy)
    ax[0].set_xlim(0.6, 5.0)
    ax[0].set_ylim(30., 27.5)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig('fit.png')
    #ax[0].set_ylim
    # 45693