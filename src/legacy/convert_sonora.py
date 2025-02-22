from astropy.table import Table, Column
import numpy as np
import astropy.units as u
import astropy.constants as c
import glob
from typing import Tuple, List, Dict
import os
from tqdm import tqdm
import shutil
from pathlib import Path
from matplotlib import pyplot as plt
from specutils import Spectrum1D
from specutils.manipulation import LinearInterpolatedResampler, FluxConservingResampler
from scipy.optimize import curve_fit
import requests
from joblib import Parallel, delayed
import tarfile
import gzip 
import h5py as h5
from numpy.lib.recfunctions import structured_to_unstructured

bands = ["f090W", "f115W", "f150W", "f200W", "f277W","f335M", "f356W", "f410M", "f444W"]
hst_bands = ["f435W", "f606W", "f814W", "f105W", "f125W", "f140W", "f160W"]
miri_bands = ["f560W", "f770W", "f1000W", "f1130W", "f1280W", "f1500W", "f1800W", "f2100W", "f2550W"]

phot_bands_all = ["f435W", 'f606W', 'f814W', "f090W", "f115W", "f150W", "f200W", "f277W","f335M", "f356W", "f410M", "f444W", "f560W", "f770W", "f1000W", "f1130W", "f1280W", "f1500W", "f1800W", "f2100W", "f2550W"]

band_wavs = {"f435W": 4_340., 'f606W':5960, 'f814W':8073,  "f090W": 9_044., 
                    "f115W": 11_571., "f150W": 15_040., "f200W": 19_934., "f277W": 27_695., 
                    "f335M": 33_639., "f356W": 35_768., "f410M": 40_844., "f444W": 44_159.,
                    'f560W':55870.25, 'f770W':75224.94, 'f1000W': 98793.45, 'f1130W':112960.71, 
                    'f1280W':127059.68,  'f1500W':149257.07,  'f1800W':178734.17, 'f2100W':205601.06, 
                    'f2550W':251515.99}

band_wavs = {key: value * u.Angstrom for (key, value) in band_wavs.items()} # convert each individual value to Angstrom
band_wavs_upper = {key.replace('f', 'F'): value for key, value in band_wavs.items()}
band_wavs.update(band_wavs_upper)

file_urls={'bobcat':["https://zenodo.org/records/5063476/files/spectra_m+0.0.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m-0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co1.5_g1000nc.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co0.5_g1000nc.tar.gz?download=1"],
           'cholla':["https://zenodo.org/records/4450269/files/spectra.tar.gz?download=1"],
           'evolution_and_photometry':["https://zenodo.org/records/5063476/files/evolution_and_photometery.tar.gz?download=1"],
           'elf_owl':['https://zenodo.org/records/12735103/files/spectra.zip?download=1'],
           'diamondback':['https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz?download=1'],
           'low-z':['https://dataverse.harvard.edu/api/access/datafile/4571308', 'https://dataverse.harvard.edu/api/access/datafile/4570758']}
def setup_libraries(path='sonora_data/'):
    for library in ["bobcat", "cholla", "evolution_and_photometry"]:
        # Ensure the destination folder exists, or create it if not
        new_path = path + f"/{library}/"
  
        os.makedirs(new_path, exist_ok=True)

        # Fetch the Zenodo metadata for the given repository URL
        for file_url in file_urls[library]:
            file_name = file_url.split("/")[-1].replace("?download=1", "")
            
            local_file_path = os.path.join(new_path, file_name)
            if not Path(local_file_path).is_file():
                print(f'Sonora {library} files not found. Downloading from Zenodo.')
                # Download the file
                response = requests.get(file_url, stream=True)
                if response.status_code == 200:
                    with open(local_file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f"Downloaded: {file_name}")
                else:
                    print(f"Failed to download: {file_name} ({response.status_code})")

        print(f"Sonora {library} files found.")

        # Unpack the downloaded files

        files = glob.glob(new_path+'*.tar.gz')

        for file in files:
            out_dir = file[:-7]
            if not Path(out_dir+'/').is_dir():
          
                tar = tarfile.open(file) 
                # extracting file 
                os.makedirs(out_dir+'/', exist_ok=True)
                tar.extractall(out_dir) 
                tar.close() 

            enclosed_files = glob.glob(out_dir+'/*') + glob.glob(out_dir+'/*/*')
            for file in enclosed_files:
                if file.endswith('.gz') and not Path(file[:-3]).is_file():
                    with gzip.open(file, 'rb') as f_in, open(file[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
def convolve_sed_v2(mags, wavs_um, filters, input='mag'):

    mask = np.isfinite(mags) & np.isfinite(wavs_um)
    len_wav = len(wavs_um)
    mags = mags[mask]
    wavs_um = wavs_um[mask]

    if len(wavs_um) < len_wav:
        print('Removed nans from input SED')
    if len(mags) != len(wavs_um) != len(filters):
        print('Inputs are different lengths!')
    if type(wavs_um[0]) == float:
        print('Assuming micronsss')
        wavs_um = [i*u.um for i in wavs_um]

    mag_band = []
    for filt in filters:
        if filt in bands:
            path = "/nvme/scratch/work/tharvey/jwst_filters/nircam/"
            unit = u.micron
            delimiter = ' '
        elif filt in hst_bands:
            path = "/nvme/scratch/work/tharvey/jwst_filters/hst/"
            unit = u.angstrom
            delimiter = ' '
        elif filt in miri_bands:
            path = '/nvme/scratch/work/tharvey/jwst_filters/miri/'
            unit = u.angstrom
            delimiter = ' '
        wav, trans = np.transpose(np.loadtxt(f"{path}F{filt[1:]}.dat", delimiter=delimiter, dtype=float, skiprows=1))
        wav = wav * unit
        wav = wav.to('micron')
       
        trap_object = lambda x: np.interp(x, wav, trans)
        max_wav = np.max(wav)
        min_wav = np.min(wav)

        #print('Filter min_wav, max_wav', filt, min_wav, max_wav)

        mask = (wavs_um < max_wav) & (wavs_um > min_wav)
        wavs_um_loop = wavs_um[mask]
        mags_loop = mags[mask]

        if len(wavs_um_loop) == 0:
            print(f'Warning! No overlap between filter {filt} and SED. Filling with 99')
            mag_bd = 99 * u.ABmag
        else:
            fluxes = mags_loop.to(u.Jy)
            trans_ob = trap_object(wavs_um_loop)
            # Photon counting convention : flambda = int(transmission * flux * lambda) / int(transmission * lambda)
            trans_int = np.trapz(trans*wav, wav)
            #print(trans_ob*fluxes, wavs_um_loop)
            top_int = np.trapz(trans_ob * fluxes * wavs_um_loop, wavs_um_loop)
            #print(top_int)
            flux = top_int / trans_int
            
            if input == 'flux':
                mag_bd = flux.to(mags.unit)
            elif input == 'mag':
                mag_bd = flux.to(u.ABmag)

        mag_band.append(mag_bd)
    return mag_band


def build_sonora_template_grid(bands, model_versions=['bobcat', 'cholla'], sonora_path='/nvme/scratch/work/tharvey/brown_dwarfs/sonora_model/', overwrite=False, output_file_name = 'photometry_grid.hdf5'):
    
    exists = False
    if os.path.isfile(sonora_path+output_file_name) and not overwrite:
        exists = True
        file = h5.read(path+output_file_name, 'r')
        model_bands = file.attrs['bands']
        # Check all bands are present
        for band in bands:
            if band not in model_bands:
                exists = False
    if exists:
        print('Sonora template grid already exists. Skipping.')
    else:
        print('Building sonora template grid.')
        
    models_table = Table()
    sonora_names = []

    for model_version in model_versions:
        sonora = Table.read(f'{sonora_path}/sonora_{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
        for pos, row in enumerate(sonora):
            name = row["path"].split("/")[-1]

            sonora_names.append(name)
            table = Table.read(sonora_path+name, names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
            table['flux_njy'] = table['flux_nu'].to(u.nJy)/1e17
            convolved_fluxes = [i.value for i in convolve_sed_v2(table['flux_njy'], table['wav'].to(u.um), bands, input='flux')]
            assert len(convolved_fluxes) == len(bands), f'Convolved fluxes and bands are different lengths: {len(convolved_fluxes)} and {len(bands)}'
            flux_column = Column(convolved_fluxes, name=name, unit=u.nJy)
            models_table.add_column(flux_column)

    template_grid = models_table.as_array()

    template_grid = structured_to_unstructured(template_grid, dtype=np.float32)

    return template_grid
        


def fit_sonora(catalog_path=None, rerun=False, mixed_cat=False, model_version='bobcat',
                sonora_path='/nvme/scratch/work/tharvey/brown_dwarfs/', id_col='#ID', 
                fieldname_col='FIELDNAME', n_jobs=6, save_output=True, plot_all=False, 
                plot=False, compare_galaxy_chi2=False, compare_galaxy_chi2_dif=4, 
                absolute_chi2_good_fit=10, fwhm_band_size=4.84, size_compare_band='f444W',
                force_fit=False, ignore_masked=False):
    '''
    Fits sonora brown dwarf templates to a catalog.
    
    Inputs: 
    catalog_path - string or astropy.table.Table instance - catalog for fitting. Default is None.
    rerun - bool - whether to rerun fitting if output columns found in catalog. 
        If rerun is False and columns are matched, fitting is not performed but 
        candidates will still be recomputed. Default is False.
    mixed_cat - bool - whether the catalog contains multiple fields. If True,
        fieldname_col must be specified and a new unique id column will be created. Default is False.
    model_version - string - which version of the sonora models to use. 'bobcat' or 'cholla'. Default is 'bobcat'.
    sonora_path - string - absolute path to sonora models. Default is '/nvme/scratch/work/tharvey/brown_dwarfs/'
    id_col - string - column name of source id in catalog. Default is '#ID'.
    fieldname_col - string - column name of field name in catalog
        if catalog contains multiple fields. Default is 'FIELDNAME'.
    n_jobs - int - number of parallel jobs to run. Default is 6.
    save_output - bool - whether to save output to catalog path or return catalog object.
        Default is True.
    plot_all - bool - whether to plot all fits. Default is False
    plot - bool - whether to plot fits with chi2 < absolute_chi2_good_fit. Default is False.
    compare_galaxy_chi2 - string - column name of chi2 from galaxy fitting to compare to. 
        If False, no comparison is made. Default is False.
    compare_galaxy_chi2_dif - float - chi2 difference between best BD fit and galaxy 
        fit to be considered a possible BD. Default is 4.
    absolute_chi2_good_fit - float - chi2 value below which a fit is
         considered good. Default is 40.
    fwhm_band_size - float - size of source in fwhm of band to be 
        considered a point source. Default is 4.84 pixels.
    size_compare_band - string - band to use for size comparison. Default is f444W.
    force_fit - bool - whether to force fitting even if source is >> point source. Default is False.
    ignore_masked - bool - whether to ignore masked fluxes for bands that are masked. Default is False.

    Returns:

    catalog - astropy.table.Table instance - catalog with new columns added.
    Columns added are as follows:
        best_template_sonora_{model_version} - string - name of best fit template.
        chi2_best_sonora_{model_version} - float - chi2 of best fit template.
        constant_best_sonora_{model_version} - float - scaling constant of best fit template.
        delta_chi2_{compare_galaxy_chi2}_sonora_{model_version} - float - chi2 difference between best BD fit and galaxy fit if compare_galaxy_chi2 is not False.
        possible_brown_dwarf_chi2_sonora_{model_version} - bool - whether source is a possible BD based on good chi2.
        possible_brown_dwarf_chi2_sonora_{model_version}_compare - bool - whether source is a possible BD based on good chi2 and chi2 difference to galaxy fit.
        possible_brown_dwarf_compact_{model_version}_compare - bool - whether source is a possible BD based on good chi2, chi2 difference to galaxy fit and size.
        

    '''

    sonora = Table.read(sonora_path+f'sonora_model/sonora_{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
    
    if type(catalog_path) == str:
        catalog = Table.read(catalog_path)
    elif type(catalog_path) == Table:
        catalog = catalog_path 

    if mixed_cat:
        field = catalog[fieldname_col]
        id = catalog[id_col]
        unique_id = [f"{i}_{f}" for i, f in zip(id, field)]
        catalog['unique_id'] = unique_id
        id_col = 'unique_id'
    
    bands = {}
    bands_all = []
    for row in catalog:
        bands[row[id_col]] = []
        for band in phot_bands_all:
            try:
                flux = row[f'FLUX_APER_{band}_aper_corr_Jy']
                if row[f'unmasked_{band}'] or ignore_masked:
                    if flux[0] < 1e15: # Avoids the merged 1e20 filler values
                        bands[row[id_col]].append(band)
                if band not in bands_all:
                    bands_all.append(band)
            except KeyError as e:
                try:
                    flux = row[f'FLUX_APER_{band.replace("f", "F")}_aper_corr_Jy']
                    if row[f'unmasked_{band.replace("f", "F")}'] or ignore_masked:
                        if flux[0] < 1e15: # Avoids the merged 1e20 filler values
                            bands[row[id_col]].append(band.replace('f', 'F'))
                            size_compare_band = size_compare_band.replace('f', 'F')
                    if band not in bands_all:
                        bands_all.append(band)
                except KeyError as e:
                    pass

    
    models_table = Table()
    models_table['band'] = bands_all
    sonora_names = []
    print(bands_all)
    # Read this file in and add if needed to the table - will be quicker than creating each time
    for pos, row in enumerate(sonora):
        name = row["path"].split("/")[-1]
        sonora_names.append(name)
        table = Table.read(sonora_path+row['path'], names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        table['flux_njy'] = table['flux_nu'].to(u.nJy)/1e17
        convolved_fluxes = [i.value for i in convolve_sed_v2(table['flux_njy'], table['wav'].to(u.um), bands_all, input='flux')]
        #print(bands_all)
        #print(convolved_fluxes)
        flux_column = Column(convolved_fluxes, name=name, unit=u.nJy)
        models_table.add_column(flux_column)
    
    #models_table.write(sonora_path+f'sonora_model/sonora_{model_version}.ecsv', overwrite=True)

    if f'best_template_sonora_{model_version}' not in catalog.colnames or rerun:
        print('Rerunning fitting.')
        params = [(id, bands_id, catalog, id_col, models_table, sonora_names, band_wavs, bands_all, plot_all, plot, sonora_path, fwhm_band_size, size_compare_band, force_fit, absolute_chi2_good_fit) for id, bands_id in bands.items()]
        output = Parallel(n_jobs=n_jobs)(delayed(parallel_sonora_fitting)(param) for param in tqdm(params, desc='Fitting BD templates'))
        output = np.array(output)
        # Save results
        catalog[f'best_template_sonora_{model_version}'] = output[:, 0]
        catalog[f'chi2_best_sonora_{model_version}'] = np.array(output[:, 1], dtype=np.float32)
        catalog[f'constant_best_sonora_{model_version}'] = np.array(output[:, 2], dtype=np.float32)/1e17
        print(catalog[f'best_template_sonora_{model_version}'])
        print(catalog[f'chi2_best_sonora_{model_version}'])
        print(catalog[f'constant_best_sonora_{model_version}'])
    else:
        print('Skipping brown dwarf fitting, columns found in catalog.')

    if compare_galaxy_chi2 != False:
        catalog[f'delta_chi2_{compare_galaxy_chi2}_sonora_{model_version}'] = catalog[f'chi2_best_sonora_{model_version}'].astype('>f4') - catalog[compare_galaxy_chi2] 
        #print('len of < 0 delta chi2')
        #print(len(catalog[f'delta_chi2_{compare_galaxy_chi2}_sonora_{model_version}'][catalog[f'chi2_best_sonora_{model_version}'] <= 0]))
        # Calculate chi2 difference between best BD fit and galaxy fit
        catalog[f'delta_chi2_{compare_galaxy_chi2}_sonora_{model_version}'][catalog[f'chi2_best_sonora_{model_version}'] <= 0] = 9999
        # Calculate likely BD candidates based on good chi2 (also filter out << hot pixel PSF sources)
        catalog[f'possible_brown_dwarf_chi2_sonora_{model_version}'] = (catalog[f'FLUX_RADIUS_{size_compare_band}'] > 1.5) & (catalog[f'chi2_best_sonora_{model_version}'] < absolute_chi2_good_fit)
        # Calculate likely BD candidates based on chi2 difference and size
        catalog[f'possible_brown_dwarf_compact_{model_version}'] = (catalog[f'possible_brown_dwarf_chi2_sonora_{model_version}'] == True) & (catalog[f'FLUX_RADIUS_{size_compare_band}'] < fwhm_band_size) 
        # Calculate likely BD candidates based on chi2 difference (also filter out << hot pixel PSF sources)
        catalog[f'possible_brown_dwarf_chi2_sonora_{model_version}_compare'] = (catalog[f'delta_chi2_{compare_galaxy_chi2}_sonora_{model_version}'] < compare_galaxy_chi2_dif)  & (catalog[f'FLUX_RADIUS_{size_compare_band}'] > 1.5) 
        # Calculate likely BD candidates based on chi2 difference and size 
        catalog[f'possible_brown_dwarf_compact_{model_version}_compare'] = (catalog[f'possible_brown_dwarf_chi2_sonora_{model_version}_compare'] == True) & (catalog[f'FLUX_RADIUS_{size_compare_band}'] < fwhm_band_size) 
        
        
        # Currently not used - calculates params (e.g. distance) based on normalization of best fit template. Only currently works in cases where a direct matches in terms of temp, logg and metallicity are found in the evolution tables.
        
        #for row in catalog[catalog[f'possible_brown_dwarf_compact_{model_version}']]:
        #    calculate_params(row[f'best_template_sonora_{model_version}'], row[f'constant_best_sonora_{model_version}'], model_version=model_version, sonora_path=sonora_path)

    #results_table = Table([list(bands.keys()), templates, chi2, constants], names=['id', 'template', 'chi2', 'constant'])
    
    if save_output:
        print('Finished fitting. Saving results.')
        catalog.write(f'{catalog_path[:-5]}_sonora_{model_version}.fits', overwrite=True)
        print('Written output to ', f'{catalog_path[:-5]}_sonora_{model_version}.fits')
    else:
        print('Finished fitting. Returning catalog')
        return catalog

def calculate_params(best_fit_model, normalization, sonora_path='sonora_model/', model_version='bobcat'):
    # Reme
    if type(normalization) == str:
        normalization = float(normalization)

  
    model = f'{sonora_path}{best_fit_model}'
    table = Table.read(model, names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])

    teff = float(table.meta['Teff'])
    c_o = float(table.meta['C/O'])
    kzz = float(table.meta['Kzz'])
    grav = np.log10(float(table.meta['grav']))
   
    metallicity = float(table.meta['[Fe/H]'])
    y = float(table.meta['Y'])
    sign = '+' if metallicity >= 0 else ''
    path_base = f'{sonora_path}/evolution_and_photometery/evolution_and_photometery/evolution_tables/evo_tables{sign}{metallicity:.1f}/nc{sign}{metallicity:.1f}_co1.0'
    mass_age  = f'{path_base}_mass_age'
    mass = f'{path_base}_mass'
    age = f'{path_base}_age'
    lbol = f'{path_base}_lbol'
    # find mass and age from teff and gravity
    # In fixed mass/age table not all log-g's and teffs are present
    table = Table.read(mass_age, format='ascii.no_header', data_start=2, guess=False, delimiter='\s')
    [table.rename_column(f'col{pos+1}', name) for pos, name in enumerate(['Teff','log g', 'Mass','Radius','log L', 'log age'])]

    mask = (table['Teff'] == teff) & (table['log g'] == grav)
    #print(table['Teff'], teff, table['log g'], grav)
    print(best_fit_model)
    if len(table[mask]) == 0:
        print('no match found')
        print('teff', teff, 'grav', grav)
    elif len(table[mask]) > 1:
        print('multiple matches found')
    else:
        radius = table[mask]['Radius'][0] * u.Rsun
            
        distance = radius/np.sqrt(normalization)
        distance = distance.to(u.kpc)
        print('match found')
        
        print('teff', teff, 'grav', grav)

        print(distance)
        #print(table.colnames)
    # meta: {C/O: '1.00', Kzz: '1.0000E+05', Teff: '200.', Y: '0.28', '[Fe/H]': '0.50', f_hole: '1.00', f_rain: '0.00', grav: '10.'}
    # read first line of model - get temp, logg, metallicity (need to copy across)
    # read in evolution file
    # find radius by matching temp, logg, metallicity
    # scale normalization
    # Remeber factor of 1e-17 in fluxes
    

def parallel_sonora_fitting(params):
    id, bands_id, catalog, id_col, models_table, sonora_names, band_wavs, bands_all, plot_all, plot, sonora_path,  fwhm_band_size, size_compare_band, force_fit, absolute_chi2_good_fit = params
    #print('Using bands: ', bands_id)
    if len(bands_id) == 0:
        #raise Exception(f'no bands found for {id}')
        pass #when running on full catalogue, some things are masked in all bands
    mask = catalog[id_col] == id
    if len(catalog[mask]) > 1:
        raise Exception(f'multiple entries for {id}')

    if not force_fit:
        if catalog[f'FLUX_RADIUS_{size_compare_band}'][mask][0] > fwhm_band_size:
            return [f'>> PSF in {size_compare_band}', -1, 0]
    
    flux = [catalog[f'FLUX_APER_{band}_aper_corr_Jy'][mask][0][0] for band in bands_id] * u.Jy
    flux = flux.to(u.nJy)
    flux_err = [catalog[f'FLUXERR_APER_{band}_loc_depth_10pc_Jy'][mask][0][0] for band in bands_id] * u.Jy
    flux_err = flux_err.to(u.nJy)
    if len(flux) == 0:
        return ['No bands', -1, 0]
    consts = []
    fits = []
    bands_id_lower = [band.replace('F', 'f') for band in bands_id]

    for sonora_name in sonora_names: 
        models_mask = [pos for pos, band in enumerate(bands_all) if band in bands_id_lower]
        wavs = [band_wavs[band].value for band in bands_id_lower]
        popt, pcov = curve_fit(lambda x, a: a * models_table[sonora_name][models_mask], wavs, flux, sigma=flux_err, p0=1e-4)
        const = popt[0]
        chi_squared = np.sum(((const * models_table[sonora_name][models_mask] - flux) / flux_err)**2)
        consts.append(const)
        fits.append(chi_squared)

    
    best_fit = np.argmin(fits)
    
    if plot_all or chi_squared < absolute_chi2_good_fit and plot:
        fig, ax = plt.subplots()
        name = sonora_names[best_fit]
        table = Table.read(f'{sonora_path}/sonora_model/{name}', names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        table['flux_njy'] = table['flux_nu'].to(u.nJy)/1e17
        ax.plot(table['wav'], table['flux_njy'] * consts[best_fit], label=name, zorder=1)
        for band in bands_id:
            ax.errorbar(band_wavs[band], flux[bands_id.index(band)], yerr=flux_err[bands_id.index(band)], fmt='o', c = 'black', zorder=2)
            ax.scatter(band_wavs[band], consts[best_fit] * models_table[name][models_table['band'] == band.replace('F', 'f')], label=name, marker='x', color='red', zorder=4)
            
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux (nJy)')
        ax.set_title(f'id: {id}, template: {name}, chi2: {fits[best_fit]:.2f}')
        fig.savefig(f'results/{id}_{name}.png', dpi=100)
        plt.close()
        #print(f'{id}, best fit: ', sonora_names[best_fit], 'with chi squared of ', fits[best_fit], 'and a constant of ', consts[best_fit])
    
     # name of best fit template, chi2 of best fit template and scaling constant of best fit template
    return sonora_names[best_fit], fits[best_fit],consts[best_fit]

def plot_sonora(path='sonora_model/', model_version='bobcat'):

    files = glob.glob(path+'*.dat')
    for file in files:
        table = Table.read(file, format='ascii.commented_header')
        table['fnu'] = table['flux_lambda'] * table['wav']**2 / c.c
        plt.plot(table['wav'], table['fnu'], label=file.split('/')[-1])
        #plt.legend()
        plt.savefig(f'{path}/sonora_{model_version}_seds.png')

def choose_sonora(out_folder='sonora_model/', model_version='bobcat', overwrite=False):
    if model_version == 'bobcat':
        temp = [200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
        log_gs = [10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160]
        m = ['-0.5', '0.0', '+0.5']
    elif model_version == 'cholla':
        temp = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
        log_gs = [31, 56, 100, 178, 316, 562, 1000, 1780, 3162]
        kzzs = [2, 4, 7]
    count = 1
    if Path(out_folder+f'sonora_{model_version}.param').is_file() and not overwrite:
        print('Sonora param file already exists. Skipping.')
        return
    
    with open(out_folder+f'sonora_{model_version}.param', 'w') as f:
        for temp in temp:
            for log_g in log_gs:
                if model_version == 'bobcat':
                    for metallicity in m:
                        
                        mname = f'_m+{metallicity}/spectra' if metallicity == '0.0' else f'_m{metallicity}'
                        
                        path = f'sonora_data/{model_version}/spectra{mname}'
                        name = f'sp_t{temp}g{log_g}nc_m{metallicity}.dat'
                        name_new = f'sp_t{temp}g{log_g}nc_m{metallicity}_resample.dat'
                        if not Path(f'{path}/{name_new}').is_file() or overwrite:
                            print(f'Making sonora file for {path}/{name[:-4]}')
                            done = convert_sonora(f'{path}/{name[:-4]}', model_version=model_version)
                            if not done:
                                print(f'Failed {name}')
                                continue 
                        f.writelines(f'{count} {out_folder}{name_new} 1.0\n')
                        shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{name_new}')
                        count += 1
                        # The special C/O ratio files
                        if log_g == 1000 and metallicity == '0.0':
                            for co in [0.5, 1.5]:
                                mname = f"_m+{metallicity}_co{co}_g{log_g}nc"
                                path = f'sonora_data/{model_version}/spectra{mname}'
                                name = f'sp_t{temp}g{log_g}nc_m+{metallicity}_co{co}.dat'
                                name_new = f'sp_t{temp}g{log_g}nc_m+{metallicity}_co{co}_resample.dat'
                                if not Path(f'{path}/{name_new}').is_file() or overwrite:
                                    print(f'Making sonora file for {path}/{name[:-4]}')
                                    done = convert_sonora(f'{path}/{name[:-4]}', model_version=model_version)
                                    if not done:
                                        continue 
                                f.writelines(f'{count} {out_folder}{name_new} 1.0\n')
                                shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{name_new}')
                                count += 1
                elif model_version == 'cholla':
                    for kzz in kzzs:
                        name = f'{temp}K_{log_g}g_logkzz{kzz}.spec'
                        path = f'sonora_data/{model_version}/spectra/spectra_files/'
                        name_new = f'{temp}K_{log_g}g_logkzz{kzz}_resample.dat'
                        if not Path(f'{path}/{name_new}').is_file() or overwrite:
                            print(f'Making sonora file for {path}/{name[:-5]}')
                            done = convert_sonora(f'{path}/{name[:-5]}', model_version=model_version)
                            
                            if not done:
                                continue 
                            f.writelines(f'{count} {out_folder}{name_new} 1.0\n')
                            shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{name_new}')
                            count += 1

def convert_sonora(path, model_version='bobcat'):
    try:
        if model_version == 'bobcat':
            with(open(path, 'r')) as f:
                header = f.readlines()[0]
                table = Table.read(path, format='ascii', data_start=2, header_start=None, guess=False, fast_reader=False, names=['microns', 'Flux (erg/cm^2/s/Hz)'], units=[u.micron, u.erg/(u.cm**2*u.s*u.Hz)])
        elif model_version == 'cholla':
            table = Table.read(path+'.spec', format='ascii', data_start=2, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
            table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
            header = ''

    except FileNotFoundError as e:
        print(e)
        return False
    table['microns'] = table['microns']
    table['wav'] = table['microns'].to(u.AA)
    table.sort('wav') 
    table_order = table['wav', 'Flux (erg/cm^2/s/Hz)']
    table_order = table_order[(table['wav'] > 0.3*u.micron) & (table['wav'] < 8*u.micron)]
    spec = Spectrum1D(spectral_axis=u.Quantity(table_order['wav']), flux=u.Quantity(table_order['Flux (erg/cm^2/s/Hz)']))
    resample = FluxConservingResampler()
    new_disp_grid = np.arange(table_order['wav'].min(), table_order['wav'].max(), 50) * u.AA
    new_spec = resample(spec, new_disp_grid)
    new_table = Table()
    new_table['wav'] = new_spec.spectral_axis
    new_table['flux_nu'] = new_spec.flux
    header = [i.replace(',', '') for i in header.split(' ') if i not in ['', '\n', ' ', '(MKS),']]
    new_table.meta = {header[i+len(header)//2]:header[i] for i in range(len(header)//2)}
    new_table.write(path+'_resample.dat', format='ascii.ecsv', overwrite=True)
    print(path+'_resample.dat')
    return True


def deduplicate_templates(
    template_grid: np.ndarray,
    tolerance: float = 0.01,
    relative: bool = True
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Remove duplicate templates from a grid based on photometric similarity.
    
    Args:
        template_grid: numpy array of shape (N, M) where N is number of templates
                      and M is number of photometric datapoints
        tolerance: maximum allowed difference between templates to be considered duplicates
        relative: if True, use relative differences (template1/template2 - 1),
                 if False, use absolute differences
    
    Returns:
        Tuple containing:
        - Deduplicated template grid
        - Dictionary mapping kept template indexes to lists of removed duplicate indexes
    """
    N, M = template_grid.shape
    # Keep track of which templates to keep and their duplicates
    to_keep = np.ones(N, dtype=bool)
    duplicate_map = {}
    
    # Compare each template with all subsequent templates
    for i in range(N):
        if not to_keep[i]:
            continue
            
        template1 = template_grid[i]
        duplicates = []
        
        for j in range(i + 1, N):
            if not to_keep[j]:
                continue
                
            template2 = template_grid[j]
            
            # Calculate differences based on method
            if relative:
                # Avoid division by zero
                safe_template2 = np.where(template2 != 0, template2, np.inf)
                diffs = np.abs(template1/safe_template2 - 1)
            else:
                diffs = np.abs(template1 - template2)
            
            # Check if all differences are within tolerance
            if np.all(diffs <= tolerance):
                to_keep[j] = False
                duplicates.append(j)
        
        if duplicates:
            duplicate_map[i] = duplicates
    
    # Create deduplicated grid
    deduplicated_grid = template_grid[to_keep]
    
    return deduplicated_grid, duplicate_map



if __name__ == '__main__':
    # Run this the first time to download and unpack the sonora files
    setup_libraries('sonora_data/')
    
    #choose_sonora(model_version='cholla', overwrite=False)
    #choose_sonora(model_version='bobcat', overwrite=False)

    #catalog_path  = '/nvme/scratch/work/tharvey/proposals/NEPAGN/NEP_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV_sonora'
    #catalog_path = '/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/ACS_WFC+NIRCam/Combined/CEERS_MASTER_Sel-f277W+f356W+f444W_v9_loc_depth_masked_10pc_EAZY_matched_selection_ext_src_UV_updated'
    
    paths = ['/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v11/ACS_WFC+NIRCam/JOF/JOF_MASTER_Sel-F277W+F356W+F444W_v11.fits',
            '/raid/scratch/work/austind/GALFIND_WORK/Catalogues/v9/NIRCam/JADES-Deep-GS+JEMS/JADES-Deep-GS+JEMS_MASTER_Sel-F277W+F356W+F444W_v9.fits']

    for field in ['CEERS_3', 'NEP_4']:
        print(f'Doing {field}')
        catalog_path = f'/nvme/scratch/work/tharvey/catalogs/shay_bd_fitting/{field}_extras'
        ext = '_zfree' if field in ['JADES'] else ''
        catalog = Table.read(catalog_path+'.fits')

        #Filter cat
        #catalog = catalog[(catalog['FLUX_RADIUS_f444W'] < 4.84) & (catalog['MAG_APER_f444W_aper_corr'][:, 0] < 28.5) & (catalog['unmasked_blank_NIRCam'] == True)]
        #catalog = catalog[catalog['FIELDNAME'] == 'CEERSP3']
        #catalog = catalog[catalog['NUMBER'] == 246]
        #catalog = catalog[catalog['selected_gal_all_criteria_delta_chi2_4_fsps_larson']== True]
        #catalog_path = 'jades_hainline_matched'
        model_versions = ['bobcat', 'cholla']
        for model_version in model_versions:
            rerun = True
            if not Path(catalog_path+f'_sonora.fits').is_file() or rerun:
                catalog = fit_sonora(catalog, mixed_cat=True if field in ['CEERS', 'NEP'] else False, model_version=model_version, compare_galaxy_chi2=f'chi2_best_fsps_larson{ext}' , save_output=False, plot_all=True, id_col='NUMBER', rerun=True, ignore_masked=True, force_fit= True)
        
        catalog.write(catalog_path+f'_sonora.fits', overwrite=True)
                #catalog.write(catalog_path+'_sonora.fits', overwrite=True)
            #catalog.write('/nvme/scratch/work/tharvey/catalogs/wang2023_bd_P3_246_sonora.fits', overwrite=True)
            
    '''catalog = Table.read(catalog+f'_sonora_{model_version}.fits')

    for row in catalog[catalog[f'possible_brown_dwarf_compact_{model_version}']]:
        print(row['NUMBER'])
        calculate_params(row[f'best_template_sonora_{model_version}'], row[f'constant_best_sonora_{model_version}'], model_version=model_version, sonora_path='sonora_model/')
    '''

    #catalog = Table.read(catalog_path)
    #print(catalog.colnames)