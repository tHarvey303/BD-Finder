import fnmatch
import glob
import gzip
import operator
import os
import re
import shutil
import tarfile
import warnings
import zipfile
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import astropy.units as u
import h5py as h5
import matplotlib.patheffects as pe
import numpy as np
import requests
import spectres
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import ascii
from astropy.table import Column, Table

# supress astropy runtime warnings
from astropy.utils.exceptions import AstropyWarning
from astroquery.svo_fps import SvoFps
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

warnings.simplefilter('ignore', category=AstropyWarning)


# Path to download models - currently supports Sonora Bobcat, Cholla, Elf Owl, Diamondback and the Low-Z models
file_urls={'sonora_bobcat':["https://zenodo.org/records/5063476/files/spectra_m+0.0.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m-0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co1.5_g1000nc.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co0.5_g1000nc.tar.gz?download=1"],
           'sonora_cholla':["https://zenodo.org/records/4450269/files/spectra.tar.gz?download=1"],
           'sonora_diamondback':['https://zenodo.org/records/12735103/files/spectra.zip?download=1'],
           'sonora_elf_owl':['https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz?download=1', #Y-type
                            'https://zenodo.org/records/10385821/files/output_1000.0_1200.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_575.0_650.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_850.0_950.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_700.0_800.tar.gz?download=1', # T-type
                            'https://zenodo.org/records/10381250/files/output_275.0_325.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_350.0_400.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_425.0_475.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_500.0_550.0.tar.gz?download=1'], #L-type
           'low-z':['https://dataverse.harvard.edu/api/access/datafile/4571308', 'https://dataverse.harvard.edu/api/access/datafile/4570758'],
           'ATMO2020':['https://perso.ens-lyon.fr/isabelle.baraffe/ATMO2020/ATMO_2020_models.tar.gz']}

evolution_tables = {'sonora_bobcat':["https://zenodo.org/records/5063476/files/evolution_and_photometery.tar.gz?download=1"],
                    'sonora_diamondback':['https://zenodo.org/records/12735103/files/evolution.zip?download=1']}

# in micron
model_wavelength_ranges = {'sonora_elf_owl':(0.6, 15),
                            'sonora_diamondback':(0.3, 250),
                            'sonora_bobcat':(0.4, 50),
                            'sonora_cholla':(0.3, 250),
                            'low-z':(0.1, 99),
                            'ATMO2020':(0.2, 100)}
# Euclid bands
# "Y", "Blue", "J", "Red", "H", "vis"
# Euclid - NISP
# Paranal - Vista - Z, Y, J, H, Ks


model_parameters = {'sonora_bobcat':{
            'temp':[200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
            'log_g':[10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160],
            'met':['-0.5', '0.0', '+0.5'],
            'co':['', 0.5, 1.5]},
        'sonora_cholla':{
            'temp':[500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300],
            'log_g':[31, 56, 100, 178, 316, 562, 1000, 1780, 3162],
            'kzz':[2, 4, 7]},
        'sonora_elf_owl':{
            'temp':[275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400],
            'log_g':[10.0, 17.0, 31.0, 56.0, 100.0, 178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0],
            'met':[-0.5, -1.0, 0.0, 0.5, 0.7, 1.0],
            'kzz':[2.0, 4.0, 7.0, 8.0, 9.0],
            'co':[0.5, 1.0, 1.5, 2.0, 2.5]},
        'sonora_diamondback':{
            'temp':[900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
            'log_g':[31, 100, 316, 1000, 3160],
            'met':['-0.5', '0.0', '+0.5'],
            'co':[1.0],
            'f':['f1', 'f2', 'f3', 'f4', 'f8', 'nc']},
        'low-z':{
            'temp':[500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600],
            'log_g':[3.5, 4.0, 4.5, 5.0, 5.25, 6.0],
            'met':['+1.0', '+0.75', '+0.25', '+0.5', '+0.0', '-2.5', '-2.0', '-1.5', '-1.0', '-0.25', '-0.5'],
            'co':[0.85, 0.1, 0.55],
            'kzz':[-1.0, 10.0, 2.0]},
        'ATMO2020':{
            'temp':[200, 250, 300, 350, 400, 450, 500, 660, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
            'log_g':[2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            #'kzz':[10, 9, 8, 7, 6, 5, 4, 2],
            'kzz_type':['none', 'weak', 'strong'],
        },
}



model_param_names = ["temp", "log_g", "kzz", "met", "co", "f", 'kzz_type']
model_param_dtypes = {
    "temp": float,
    "log_g": float,
    "kzz": float,
    "met": str,
    "co": str,
    "f": str,
    'kzz_type': str,
}

# elf owl
# The parameters included within this grid are effective temperature (Teff), gravity (log(g)), vertical eddy diffusion coefficient (log(Kzz)), atmospheric metallicity ([M/H]), and Carbon-to-Oxygen ratio (C/O).


default_bands = ["ACS_WFC.F435W", "ACS_WFC.F475W", "ACS_WFC.F606W", "ACS_WFC.F625W", "ACS_WFC.F775W", "ACS_WFC.F814W", "ACS_WFC.F850LP",
                "F070W", "F090W", "WFC3_IR.F105W", "WFC3_IR.F110W", "F115W", "WFC3_IR.F125W", "F150W",
                "WFC3_IR.F140W", "F140M", "WFC3_IR.F160W", "F162M", "F182M", "F200W", "F210M", "F250M",
                "F277W", "F300M", "F335M", "F356W", "F360M", "F410M", "F430M",
                "F444W", "F460M", "F480M", "F560W", "F770W",
                "NISP.Y", "NISP.J", "NISP.H", "VIS.vis",
                "VISTA.Z", "VISTA.Y", "VISTA.J", "VISTA.H", "VISTA.Ks", "IRAC.I1", "IRAC.I2",
                "MegaCam.u", "MegaCam.g", "MegaCam.r", "MegaCam.i", "MegaCam.z",
                "HSC.g", "HSC.r", "HSC.i", "HSC.z", "HSC.Y"]

 # "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"]

code_dir = os.path.dirname(os.path.abspath(__file__))


class StarFit:
    def __init__(self, libraries = ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback', 'low-z', 'ATMO2020'], library_path='internal', compile_bands='default',
                parameter_overrides={'sonora_cholla':{'model':'sonora_bobcat', 'met':'0.0', 'co':''}, 'sonora_elf_owl':{'model':'sonora_bobcat', 'met':'closest', 'co':'closest'}},
                facilities_to_search={"JWST": ["NIRCam", "MIRI"], "HST": ["ACS", "WFC3"], "Euclid":["NISP", "VIS"], "Paranal":["VIRCam"], "Spitzer":["IRAC"], "CFHT":["MegaCam"], "Subaru":['HSC']}, resample_step=50, constant_R=True, R=300, min_wav=0.3 * u.micron,
                max_wav=12 * u.micron, default_scaling_factor=1e-22, verbose=False):

        '''

        Parameters
        ----------
        library_path : str
            Path to the model libraries. Default is 'models/'.
        libraries : list
            List of model libraries to compile. Default is ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback'].
        compile_bands : list
            List of bands to compile the models for. Default is 'default', which uses the default_bands.
        parameter_overrides : dict
            Dictionary of parameter overrides for specific libraries. Default is {'sonora_cholla':'sonora_bobcat'}. This is used to
            allow estimation of distances/luminosities/masses etc for models without evolutionary tables by linking them to a similar library.
            The matching is done by the model name and the parameter values are taken from the override library. I.e the default is to allow
            Cholla models to be matched to Bobcat models, where we take the M/H 0.0 and solar C/O models from Bobcat when Cholla models are used.
        facilities_to_search : dict
            Dictionary of facilities and instruments to search SVO for filters. Default is {"JWST": ["NIRCam", "MIRI"], "HST": ["ACS", "WFC3_IR"]}.
        resample_step : int
            Wavelength step in Angstrom to resample the models to. Default is 50. Only used if constant_R is False.
        constant_R : bool
            If True, the models will be resampled to a constant resolution. Default is False.
        R : int
            The resolution to resample the models to. Default is 500. Only used if constant_R is True.
        min_wav : astropy Quantity
            Minimum wavelength to include in the models. Default is 0.3 microns.
        max_wav : astropy Quantity
            Maximum wavelength to include in the models. Default is 10 microns.
        default_scaling_factor : float
            Scaling factor to shift the models to reasonable flux ranges. In general use this should not be modified, as changing this
            without also recomputing all photometry may result in incorrectly normalized SEDs, although it is also saved to the HDF5 grids. Default is 1e-22. 
        '''

        if library_path == 'internal':
            self.library_path = os.path.dirname(os.path.dirname(code_dir)) + '/models'
        else:
            self.library_path = library_path

        print(f'Library path: {self.library_path}')

        if type(libraries) is str:
            libraries = [libraries]

        self.libraries = libraries

        self.band_codes = {}
        self.filter_wavs = {}
        self.filter_ranges = {}
        self.filter_instruments = {}
        self.transmission_profiles = {}
        self.facilities_to_search = facilities_to_search
        self.model_parameters = model_parameters
        self.verbose = verbose

        self.parameter_overrides = parameter_overrides

        self.resample_step = resample_step
        self.constant_R = constant_R
        self.R = R
        self.min_wav = min_wav
        self.max_wav = max_wav
        self.default_scaling_factor = default_scaling_factor

        self.template_grids = {}
        self.template_bands = {}
        self.template_names = {}
        self.template_parameters = {}
        self.template_ranges = {}
        self.scaling_factors = {}

        if compile_bands == 'default':
            self.model_filters = default_bands
        else:
            self.model_filters = compile_bands


        self._fetch_transmission_bandpasses()

        # If compiled data exists, load it. If not, run setup_libraries.

        for library in libraries:
            if not os.path.exists(f"{self.library_path}/{library}_photometry_grid.hdf5"):
                print(f'Compiled {library} photometry not found.')
                if not os.path.exists(f'{self.library_path}/{library}/'):
                    print(f'No {library} models found. Running setup_libraries.')
                    self.setup_libraries(self.library_path, [library])
                self.convert_templates(self.library_path, library)
                self.build_template_grid(self.model_filters, model_versions=[library], model_path=self.library_path)

            self._load_template_grid(self.library_path, library)

        self.combined_libraries = None

        self._build_combined_template_grid(libraries)

        print(f'Total models: {len(self.combined_template_grid.T)}')

    def __repr__(self):
        return f"{self.__class__.__name__}({','.join(self.libraries)})"

    def param_abrev(self, param, library):
        if library == 'sonora_bobcat':
            vals = {'temp':'t', 'log_g':'g', 'met':'m', 'co':'co'}
        else:
            raise NotImplementedError(f'Library {library} not supported.')

        return vals[param]


    def select_photometry_table(self, model_idx):
        library, name = self.get_template_name(model_idx)
        parameters = self.get_template_parameters(model_idx)

        info = {}
        # This allows using e.g. sonora bobcat evolution tables for elf owl models.
        if library in self.parameter_overrides.keys():
            overrides = self.parameter_overrides.get(library, {}).copy()
            info_overrides = overrides.copy()
            new_library = overrides['model']
            for parameter, key in overrides.items():
                if 'closest' in key:
                    available_parameters = self.model_parameters[new_library][parameter]
                    if type(available_parameters[0]) is str:
                        # This removes e.g. + from metallcitiy values, and converts '' to 1 e.g. for solar C/O
                        available_parameters_corr = [float(str(i).replace('+', '')) if i != '' else 1 for i in available_parameters]
                    else:
                        available_parameters_corr = available_parameters
                    available_parameters_arr = np.array(available_parameters_corr)
                    closest_pos = np.argmin(np.abs(available_parameters_arr - float(parameters[parameter])))
                    closest = available_parameters[closest_pos]
                    #print(f'Overriding {parameter} from {parameters[parameter]} to {closest}')
                    overrides[parameter] = closest
                    info_overrides[parameter] = available_parameters_arr[closest_pos]

            info['overrides'] = info_overrides

            parameters.update(overrides)
            library = new_library

        if library == 'sonora_bobcat':
            if parameters['co'] not in ['', "b'nan'", 'nan']:
                table_name = f'flux_table_JWST_m+0.0_co{parameters["co"]}'
            else:
                met = parameters["met"]
                if met == '0.0':
                    met = '+0.0'
                table_name = f'flux_table_JWST{met}'
            #print(table_name, parameters['co'])
            path = f'{self.library_path}/{library}_evolution/evolution_and_photometery/photometry_tables/{table_name}'

            # Could use these tables instead?
            #alt_path = f'{self.library_path}/{library}_evolution/evolution_and_photometery/evolution_tables/evo_tables{met}/nc{met}_co1.0_mass_age'
            #tab_col_names = ascii.read(path, format='fixed_width', fast_reader=False, guess=False, delimiter='  ', header_start=1, data_start=2, data_end=2)
            #tab = ascii.read(path, format='basic', fast_reader=False, guess=False, delimiter='\s', header_start=None, data_start=2, names=tab_col_names.colnames)


            tab_col_names = ascii.read(path, format='fixed_width', fast_reader=False, guess=False, delimiter='  ', header_start=4, data_start=5, data_end=5)
            tab = ascii.read(path, format='basic', fast_reader=False, guess=False, delimiter=r'\s', header_start=None, data_start=5, names=tab_col_names.colnames)
            tab['R/Rsun'] = [float(str(i).replace('*', '')) for i in tab['R/Rsun']]
            # Round to avoid having 1201, 1200 be separate values. Rounds to nearest 5 which should be ok.
            tab['Teff'] = np.round(tab['Teff']*2, -1)/2

        elif library == 'ATMO2020':
            if parameters["kzz_type"] == 'none':
                kzz_type = 'CEQ'
            elif parameters["kzz_type"] == 'weak':
                kzz_type = 'NEQ_weak'
            elif parameters["kzz_type"] == 'strong':
                kzz_type = 'NEQ_strong'


            path = f'{self.library_path}/{library}/models/evolutionary_tracks/ATMO_{kzz_type}/JWST_photometry/JWST_phot_NIRCAM_modAB_mean/'
            # folder
            files = glob.glob(f'{path}/*')
            tables = []
            paths = []
            #print(f'Looking for {parameters["temp"]}, {parameters["log_g"]}, {parameters["kzz_type"]}')
            for file in files:
                tab = ascii.read(file)
                min_teff, max_teff = tab['Teff'].min(), tab['Teff'].max()
                min_logg, max_logg = tab['Gravity'].min(), tab['Gravity'].max()
                # round min and max logg to nearest 0.1
                min_logg = np.round(min_logg*10)/10
                max_logg = np.round(max_logg*10)/10

                print(f'Found {min_teff}, {max_teff}, {min_logg}, {max_logg}')

                if parameters['temp'] >= min_teff and parameters['temp'] <= max_teff and parameters['log_g'] >= min_logg and parameters['log_g'] <= max_logg:
                    tables.append(tab)
                    paths.append(file)

            tab = tables
            path = paths

        else:
            return None, None, None
        return tab, path, info


    def interpolate_stellar_parameters(self, teff, logg, mass_interp, radius_interp):
        """
        Interpolate mass and radius for given temperature and log g values
        
        Args:
            teff (float or array): Effective temperature
            logg (float or array): Surface gravity (log g)
            mass_interp: Mass interpolator
            radius_interp: Radius interpolator
            
        Returns:
            tuple: (mass, radius) - Interpolated values
        """
        # Create input points for interpolation
        if np.isscalar(teff) and np.isscalar(logg):
            points = np.array([[teff, logg]])
        else:
            # Handle array inputs
            teff = np.asarray(teff)
            logg = np.asarray(logg)
            if teff.shape != logg.shape:
                raise ValueError("teff and logg must have the same shape")

            # Reshape arrays into points for interpolation
            points = np.column_stack((teff.flatten(), logg.flatten()))

        # Interpolate values
        mass = mass_interp(points)
        radius = radius_interp(points)

        # Reshape back to original shape if needed
        if not np.isscalar(teff):
            mass = mass.reshape(teff.shape)
            radius = radius.reshape(teff.shape)

        return mass, radius


    def plot_model_photometry(self, model_idx, norm, ax, flux_unit=u.nJy, wav_unit=u.um, test_scale = 1, label=None, **kwargs):
        library, name = self.get_template_name(model_idx)
        table, path, info = self.select_photometry_table(model_idx)
        params = self.get_template_parameters(model_idx)


        if table is None:
            return np.nan, np.nan

        if library in self.parameter_overrides.keys():
            params.update(self.parameter_overrides[library])
            new_library = self.parameter_overrides[library]['model']
            library = new_library

        plotted = False

        if library == 'sonora_bobcat':

            possible_bands = [i for i in table.colnames if i not in ['Teff', 'log g', 'mass', 'R/Rsun']]

            temp = float(params['temp'])
            log_g = float(params['log_g'])

            # Find row in table that matches the model parameters (within 25 K and 0.1 dex)
            init_row = table[np.abs(table['Teff'] - temp) < 25]
            row = init_row[np.abs(init_row['log g'] - np.log10(1e2*log_g)) < 0.1]

            if len(init_row) == 0:
                #print(f"No match for {temp} within 25K, available: {np.unique(table['Teff'])}")
                return np.nan, np.nan
            if len(row) == 0:
                #print(f"No match for {np.log10(1e2*log_g)} within 0.1 dex, available: {np.unique(table['log g'])}")
                return np.nan, np.nan


            row = row[0]
            results = self.get_physical_params(model_idx, norm)
            if results is None:
                return np.nan, np.nan

            distance = results['distance']

            for band in possible_bands:
                if band in self.bands_to_fit:
                    wav = self.filter_wavs[band]
                    flux = 10 ** row[band] # mJy
                    # flux is normalized to 10 pc. Move to distance.
                    flux = flux * (10 * u.pc / distance)**2
                    flux *= u.mJy
                    ax.scatter(wav.to(wav_unit).value, test_scale*flux.to(flux_unit).value, marker='s', edgecolor='white', label=label if not plotted else '', **kwargs)
                    plotted = True

        elif library == 'ATMO2020':
            # TEMP

            return None


            results = self.get_physical_params(model_idx, norm)
            if results is None:
                return np.nan, np.nan

            # Need to deal with distances not always matching num(tables) due to not finding a match.
            distance = results['distance']

            #print(distance)

            for pos, tab in enumerate(table):
                temp = float(params['temp'])
                log_g = float(params['log_g'])
                row = tab[np.abs(tab['Teff'] - temp) < 25]
                row = row[np.abs(row['Gravity'] - log_g) < 0.1]
                if len(row) == 0:
                    continue
                elif len(row) > 1:
                    # Find closest in both
                    row = row[np.argmin(np.sqrt((row['Teff'] - temp)**2 + (row['Gravity'] - log_g)**2))]

                if type(row) is Table:
                    row = row[0]

                dis = distance[pos]

                for colname in row.colnames:
                    band = colname.split('-')[-1]

                    if band in self.bands_to_fit:
                        wav = self.filter_wavs[band]
                        mag_vega = row[colname]
                        vega_zp = self.vega_zps[band]
                        flux = 10 ** ((vega_zp - mag_vega) / 2.5) # Jy
                        flux = flux * (10 * u.pc / dis)**2
                        flux *= u.Jy
                        ax.scatter(wav.to(wav_unit).value, test_scale*flux.to(flux_unit).value, marker='s', edgecolor='white', label=label if not plotted else '', **kwargs)
                        plotted = True

        else:
            print(f'Warning! Library {library} not supported for parameter photometry plotting.')
            return None

    def get_params_from_table(self, model_idx, plot_grid=False):
        library, name = self.get_template_name(model_idx)
        table, path, info = self.select_photometry_table(model_idx)
        if table is None:
            return {}

        parameters = self.get_template_parameters(model_idx)

        if library in self.parameter_overrides.keys():
            parameters.update(self.parameter_overrides[library])
            new_library = self.parameter_overrides[library]['model']
            library = new_library

        if library == 'sonora_bobcat':
            colnames = {'temp':'Teff', 'log_g':'log g', 'mass':'mass', 'radius':'R/Rsun'}
            logg = float(parameters['log_g'])
            temp = float(parameters['temp'])

            try:
                [table.rename_column(colnames[key], key) for key in colnames.keys()]
            except:
                pass

            mass_interpolator, radius_interpolator, (temp_unique, logg_unique) = self.create_stellar_interpolator(table)

            if plot_grid:
                filename = os.path.basename(path)
                self.visualize_grid(table, temp_unique, logg_unique, mass_interpolator, radius_interpolator, plot_name=f"{library}_{filename}_mass_radius_grid.png")

            #+2 to convert from cm/s^2 to m/s^2
            mass, radius = self.interpolate_stellar_parameters(temp, np.log10(logg)+2.0, mass_interpolator, radius_interpolator)
            mass = mass[0] * u.Mjupiter
            radius = radius[0] * u.Rsun

            results = {'mass':mass, 'radius':radius}

        elif library == 'ATMO2020':
            colnames = {'temp':'Teff', 'log_g':'Gravity', 'mass':'Mass', 'radius':'Radius', 'luminosity':'Luminosity', 'age':'Age'}
            if type(table) is list:
                # Loop over tables and get estimates for each.
                results = {'mass':[], 'radius':[], 'luminosity':[], 'age':[], 'log_g':[], 'temp':[]}
                if len(table) == 0:
                    return {}

                print(f'Found {len(table)} tables for {name}')
                for tab in table:
                    try:
                        [tab.rename_column(colnames[key], key) for key in colnames.keys()]
                    except Exception as e:
                        print(e)
                        pass

                    # Find row with closest parameters
                    temp = float(parameters['temp'])
                    logg = float(parameters['log_g'])

                    row = tab[np.abs(tab['temp'] - temp) < 25]
                    row = row[np.abs(row['log_g'] - logg) < 0.1]

                    if len(row) == 0:
                        continue
                    elif len(row) > 1:
                       # Find closest in both
                        row = row[np.argmin(np.sqrt((row['temp'] - temp)**2 + (row['log_g'] - logg)**2))]

                    if type(row) is Table:
                        row = row[0]

                    units = {'mass':u.Msun, 'radius':u.Rsun, 'luminosity':u.Lsun, 'age':u.Gyr, 'log_g':u.dimensionless_unscaled, 'temp':u.K}
                    mass = row['mass']
                    radius = row['radius']
                    luminosity = 10**row['luminosity']
                    age = row['age']
                    log_g_row = row['log_g']
                    temp_row = row['temp']
                    #print(f'Mass: {mass}, Radius: {radius}, Luminosity: {luminosity}, Age: {age} for log g: {logg} (row log g: {log_g_row}), temp: {temp} (row temp: {temp_row})')
                    results['mass'].append(mass)
                    results['radius'].append(radius)
                    results['luminosity'].append(luminosity)
                    results['age'].append(age)
                    results['log_g'].append(log_g_row)
                    results['temp'].append(temp_row)

                if len(results['mass']) == 0:
                    return {}

                # Apply units
                for key in results.keys():
                    results[key] = np.array(results[key]) * units[key]


        else:
            print(f'Warning! Library {library} not supported for mass and radius estimates.')
            results = {}

        results['info'] = info

        return results


    def get_physical_params(self, idx, norm):
        results = self.get_params_from_table(idx)

        if 'mass' in results.keys() and 'radius' in results.keys():
            mass = results['mass']
            radius = results['radius']
        else:
            return None

        library, name = self.get_template_name(idx)

        # norm is R^2/d^2
        distance = np.sqrt(radius**2 / (norm*self.scaling_factors[library]))
        distance = distance.to(u.pc)

        results['distance'] = distance

        return results

    def _load_template_grid(self, library_path, library):

        with h5.File(f"{library_path}/{library}_photometry_grid.hdf5", 'r') as file:
            self.template_grids[library] = np.array(file['template_grid'][:])
            self.template_bands[library] = list(file.attrs['bands'])
            if not all(filt in self.template_bands[library] for filt in self.model_filters):
            #if self.template_bands[library] != self.model_filters:
                print(f'Warning! Model filters and compiled filters do not match for {library}.')
                print(self.model_filters, type(self.model_filters))
                print('Attempting to add:', [i for i in self.model_filters if i not in self.template_bands[library]])
                file.close()
                self.build_template_grid(self.model_filters, library, model_path=library_path)
                self._load_template_grid(library_path, library)

            scale = file.attrs['scale']
            self.scaling_factors[library] = scale

            if 'meta' in file:
                if library not in self.template_parameters.keys():
                    self.template_parameters[library] = {}
                for key in list(file['meta'].keys()):
                    skey = str(key)
                    #print(file['meta'][skey][:])
                    #print(self.template_parameters[library][skey], file['meta'][skey][:])
                    self.template_parameters[library][skey] = file['meta'][skey][:]
                    # parse string metas if needed
                    if any([isinstance(i, bytes) for i in self.template_parameters[library][skey]]):
                        self.template_parameters[library][skey] = [i.decode('utf-8') for i in self.template_parameters[library][skey]]

            if 'range' in file:
                min_wav = file['range']['min_wav'][()]
                max_wav = file['range']['max_wav'][()]
                self.template_ranges[library] = np.array([min_wav, max_wav]).T * u.um
            else:
                self.add_min_max_wavelengths_to_h5(library)

            self.template_names[library] = list(file['names'][:])
            self.template_names[library] = [i.decode('utf-8') for i in self.template_names[library]]

            file.close()

    def _build_combined_template_grid(self, libraries='all'):

        if self.combined_libraries == libraries:
            return self.combined_template_grid

        self.combined_libraries = libraries

        if libraries == 'all':
            libraries = self.libraries

        # get idx of bands in each library to match self.model_filters
        idxs = {}
        for library in libraries:
            assert set(self.model_filters).issubset(self.template_bands[library]), f"Model filters not found in {library} template bands."
            idxs[library] = np.array([self.template_bands[library].index(band) for band in self.model_filters])
            assert len(self.model_filters) == len(idxs[library]), f"Model filters and idxs are different lengths: {len(self.model_filters)} and {len(idxs[library])}"
        # get the template grid for each library
        template_grids = [self.template_grids[library] for library in libraries]
        idxs_order = [idxs[library] for library in libraries]

        # vstack the template grids in the order of self.model_filters
        self.combined_template_grid = np.hstack([template_grid[idx, :] for template_grid, idx in zip(template_grids, idxs_order)])

        # make idx range dictionary
        idx_ranges = {}
        start = 0
        for library in libraries:
            idx_ranges[library] = (start, start+len(self.template_grids[library].T))
            start += len(self.template_grids[library].T)

        self.idx_ranges = idx_ranges

        return self.combined_template_grid

    def _type_from_temp(self, temp):
        # TODO: This currently does nothing
        # From Sonora Elf Owl
        temp_range = {'Y':(275, 550), 'T':(575, 1200), 'L':(1300, 2400)}

        for sp_type, (t_min, t_max) in temp_range.items():
            if temp >= t_min and temp <= t_max:
                return sp_type

    def setup_libraries(self, path='sonora_data/', libraries = ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback', 'low-z'], urls=file_urls):
        for library in libraries:
            # Ensure the destination folder exists, or create it if not
            new_path = path + f"/{library}/"

            os.makedirs(new_path, exist_ok=True)

            # Fetch the Zenodo metadata for the given repository URL
            for file_url in urls[library]:
                if "?download=1" not in file_url:
                    file_name = 'models.tar.gz'
                else:
                    file_name = file_url.split("/")[-1].replace("?download=1", "")
                # instead get file name provided by download.

                local_file_path = os.path.join(new_path, file_name)
                if not Path(local_file_path).is_file():
                    print(f'{library} files not found. Downloading from Zenodo.')
                    # Download the file
                    response = requests.get(file_url, stream=True)
                    try:
                        file_name = response.headers['Content-Disposition'].split("attachment; filename*=UTF-8''")[-1].replace('attachment; filename=', '')
                    except:
                        file_path = file_url.split("/")[-1]

                    local_file_path = os.path.join(new_path, file_name)

                    if response.status_code == 200:
                        with open(local_file_path, "wb") as file:
                            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {file_name}", unit="KB"):
                                file.write(chunk)
                        print(f"Downloaded: {file_name}")
                    else:
                        print(f'Failed to download: {file_name} ({response.status_code})')

            print(f"{library} raw files found.")

            # Unpack the downloaded files

            files = glob.glob(new_path+'*.tar.gz') + glob.glob(new_path+'*.zip')

            for file in files:
                out_dir = file[:-7]
                if not Path(out_dir+'/').is_dir():
                    if file.endswith('.zip'):
                        with zipfile.ZipFile(file, 'r') as zip_ref:
                            zip_ref.extractall(out_dir)
                    elif file.endswith('.tar.gz'):
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

                # Fix permissions issues
                os.system(f'chmod -R 777 {out_dir}/')

    def _convolve_sed(self, mags, wavs_um, filters, input='mag'):

        self._fetch_transmission_bandpasses()

        mask = np.isfinite(mags) & np.isfinite(wavs_um)
        len_wav = len(wavs_um)
        mags = mags[mask]
        wavs_um = wavs_um[mask]

        if isinstance(wavs_um[0], float):
            wavs_um = [i * u.um for i in wavs_um]

        mag_band = []
        for filt in filters:
            unit= u.Angstrom
            wav, trans = self.transmission_profiles[filt]

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
                #print(f'Warning! No overlap between filter {filt} and SED. Filling with 99')
                if input == 'flux':
                    mag_bd = 0*mags.unit
                elif input == 'mag':
                    mag_bd = 99*u.ABmag
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

    def _fix_names(self, model_version, model_path='/nvme/scratch/work/tharvey/brown_dwarfs/models/', output_file_name='photometry_grid.hdf5'):
        model_table = Table.read(f'{model_path}/{model_version}.param', format='ascii', delimiter=' ', names=['path'])
        assert len(model_table) > 0, f'No models found in {model_version}.param. Maybe remove the file and try again.'

        names = []
        for pos, row in tqdm(enumerate(model_table), total=len(model_table), desc=f'Building {model_version} template grid'):
            name = row["path"].split("/")[-1]
            names.append(name)

        model_file_name = f'{model_version}_{output_file_name}'

        with h5.File(f'{model_path}/{model_file_name}', 'a') as file:
            file.create_dataset('names', data=names)
            file.close()

    def build_template_grid(self, bands, model_versions=['sonora_bobcat', 'sonora_cholla'], 
            model_path='/nvme/scratch/work/tharvey/brown_dwarfs/models/', overwrite=False, output_file_name='photometry_grid.hdf5'):

        if type(model_versions) == str:
            model_versions = [model_versions]

        for model_version in model_versions:
            model_file_name = f'{model_version}_{output_file_name}'
            file_path = f'{model_path}/{model_file_name}'

            # Check if file exists
            if os.path.isfile(file_path) and not overwrite:
                # Open existing file and get current bands
                with h5.File(file_path, 'r') as file:
                    existing_bands = list(file.attrs['bands'])
                    missing_bands = [band for band in bands if band not in existing_bands]

                    if not missing_bands:
                        print(f'{model_version} template grid already exists with all requested bands. Skipping.')
                        continue

                    print(f'{model_version} template grid exists but missing bands: {missing_bands}. Adding them.')

                    # Get existing data - note dimensions are (nfilters, ntemplates)
                    template_grid = file['template_grid'][:]
                    names = file.attrs.get('names', file['names'][:] if 'names' in file else [])
                    scale = file.attrs['scale']

                    # Copy meta information
                    metas = {}
                    if 'meta' in file:
                        for key in file['meta']:
                            metas[key] = file['meta'][key][:]
                            # parse string metas if needed
                            if any([isinstance(i, bytes) for i in metas[key]]):
                                metas[key] = [i.decode('utf-8') for i in metas[key]]

                file.close()

                # Read model parameters
                model_table = Table.read(f'{model_path}/{model_version}.param', format='ascii', delimiter=' ', names=['path'])
                assert len(model_table) > 0, f'No models found in {model_version}.param. Maybe remove the file and try again.'
                assert len(model_table) == template_grid.shape[1], f'Number of models ({len(model_table)}) does not match template grid shape ({template_grid.shape[1]})'

                # Create a new combined band list, preserving the order of the input bands
                combined_bands = []
                for band in bands:
                    if band not in combined_bands:
                        combined_bands.append(band)
                for band in existing_bands:
                    if band not in combined_bands:
                        combined_bands.append(band)

                # Create a new template grid with the correct shape (nfilters, ntemplates)
                new_template_grid = np.zeros((len(combined_bands), template_grid.shape[1]), dtype=np.float32)

                # Map old and new band positions
                old_to_new_indices = {i: combined_bands.index(band) for i, band in enumerate(existing_bands)}

                # Copy existing data to the correct positions in the new grid
                for old_idx, new_idx in old_to_new_indices.items():
                    new_template_grid[new_idx] = template_grid[old_idx]

                # Process missing bands
                missing_band_indices = [combined_bands.index(band) for band in missing_bands]

                for i, row in tqdm(enumerate(model_table), total=len(model_table), desc=f'Adding bands to {model_version} template grid'):
                    name = row["path"].split("/")[-1]

                    # Read model spectrum
                    table = Table.read(f'{model_path}/{row["path"]}', names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
                    table['flux_njy'] = table['flux_nu'].to(u.nJy)*scale

                    # Update meta information if not already present
                    meta_keys_tab = table.meta.keys()
                    for key in meta_keys_tab:
                        if key not in metas:
                            metas[key] = np.full(template_grid.shape[1], np.nan)


                    # Calculate fluxes for missing bands only
                    convolved_fluxes = self._convolve_sed(table['flux_njy'], table['wav'].to(u.um), missing_bands, input='flux')

                    # Place the new fluxes at the correct positions
                    for j, band_idx in enumerate(missing_band_indices):
                        new_template_grid[band_idx, i] = convolved_fluxes[j].value

                # Convert string metas if needed
                for key in metas:
                    if any([isinstance(i, str) for i in metas[key]]):
                        metas[key] = [str(i) for i in metas[key]]

                # Save to file
                with h5.File(file_path, 'w') as file:
                    file.create_dataset('template_grid', data=new_template_grid, compression='gzip')
                    file.attrs['bands'] = combined_bands
                    file.attrs['scale'] = scale

                    file.create_dataset('names', data=names)

                    # Store metadata
                    file.create_group('meta')
                    for key in metas:
                        file['meta'][key] = metas[key]

            else:
                # Original code for creating a new file
                print(f'Creating new {model_version} template grid.')
                models_table = Table()
                names = []

                model_table = Table.read(f'{model_path}/{model_version}.param', format='ascii', delimiter=' ', names=['path'])
                assert len(model_table) > 0, f'No models found in {model_version}.param. Maybe remove the file and try again.'
                metas = {}
                for pos, row in tqdm(enumerate(model_table), total=len(model_table), desc=f'Building {model_version} template grid'):
                    name = row["path"].split("/")[-1]

                    names.append(name)
                    table = Table.read(f'{model_path}/{row["path"]}', names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
                    table['flux_njy'] = table['flux_nu'].to(u.nJy)*self.default_scaling_factor

                    meta_keys_tab = table.meta.keys()
                    for key in meta_keys_tab:
                        if key not in metas.keys():
                            metas[key] = []
                            if pos > 0:
                                metas[key] = [np.nan]*pos

                    convolved_fluxes = [i.value for i in self._convolve_sed(table['flux_njy'], table['wav'].to(u.um), bands, input='flux')]
                    assert len(convolved_fluxes) == len(bands), f'Convolved fluxes and bands are different lengths: {len(convolved_fluxes)} and {len(bands)}'
                    flux_column = Column(convolved_fluxes, name=name, unit=u.nJy)

                    for key in metas.keys():
                        if key in meta_keys_tab:
                            metas[key].append(table.meta[key])
                        else:
                            metas[key].append(np.nan)

                    models_table.add_column(flux_column)

                for key in metas.keys():
                    if any([isinstance(i, str) for i in metas[key]]):
                        metas[key] = [str(i) for i in metas[key]]

                template_grid = models_table.as_array()
                template_grid = structured_to_unstructured(template_grid, dtype=np.float32)

                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                with h5.File(file_path, 'w') as file:
                    file.create_dataset('template_grid', data=template_grid, compression='gzip')
                    file.attrs['bands'] = bands
                    file.attrs['scale'] = self.default_scaling_factor

                    file.create_dataset('names', data=names)
                    # Stores meta for dataset - temperature, log_g, met, etc. Useful for filtering models, calculating distances, etc.
                    file.create_group('meta')
                    for key in metas.keys():
                        file['meta'][key] = metas[key]


            self.template_grids[model_version] = template_grid
            self.template_bands[model_version] = bands
            self.template_names[model_version] = names
            self.template_parameters[model_version] = {}
            self.add_min_max_wavelengths_to_h5(model_version)

    def _deduplicate_templates(
        self,
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

    def _fetch_transmission_bandpasses(self, save_folder: str = 'filters/'):

        if len(self.model_filters) != len(self.band_codes):
            filter_wavs = {}
            filter_instruments = {}
            filter_ranges = {}
            band_codes = {}
            vega_zps = {}

            done_band = False
            for facility in self.facilities_to_search:
                if done_band:
                    break
                for instrument in self.facilities_to_search[facility]:
                    try:
                        svo_table = SvoFps.get_filter_list(
                            facility=facility, instrument=instrument
                        )
                    except Exception as e:
                        print(e)
                        continue
                    bands_in_table = [
                        i.split(".")[-1] for i in svo_table["filterID"]
                    ]

                    bands_instruments = [i.split("/")[1] for i in svo_table["filterID"]]
                    for band in self.model_filters:
                        if band in bands_in_table or band in bands_instruments:
                            if band in filter_wavs.keys():
                                print(f'Warning! {band} found in multiple instruments. Keeping first, which is {filter_instruments[band]}. Provide instrument.band in dictionary to override this.')

                            else:
                                if band in bands_instruments:
                                    svo_band = band.split(".")[1]
                                    instrument = band.split(".")[0]

                                else:
                                    svo_band = band

                                filter_instruments[band] = instrument

                                mask = (
                                    svo_table["filterID"]
                                    == f"{facility}/{instrument}.{svo_band}"
                                )
                                assert np.sum(mask) == 1, f"Multiple or no profiles found for {facility}/{instrument}.{svo_band}"

                                wav = svo_table[mask]["WavelengthCen"]

                                upper = (
                                    svo_table[mask]["WavelengthCen"]
                                    + svo_table[mask]["FWHM"] / 2.0
                                )[0]
                                lower = (
                                    svo_table[mask]["WavelengthCen"]
                                    - svo_table[mask]["FWHM"] / 2.0
                                )[0]
                                range = (lower, upper) * wav.unit

                                if len(wav) > 1:
                                    raise Exception(
                                        f"Multiple profiles found for {band}"
                                    )

                                filter_wavs[band] = wav[0] * wav.unit

                                band_codes[band] = svo_table[mask]["filterID"][0]

                                filter_ranges[band] = range

                                if svo_table[mask]["MagSys"][0] == "Vega":
                                    vega_zps[band] = svo_table[mask]["ZeroPoint"][0]

            assert len(filter_wavs) == len(self.model_filters), f"Missing filters: {set(self.model_filters) - set(filter_wavs.keys())}"

            self.band_codes = band_codes
            self.filter_wavs = filter_wavs
            self.filter_ranges = filter_ranges
            self.filter_instruments = filter_instruments
            self.pivot = np.array([self.filter_wavs[band].to(u.AA).value for band in self.model_filters])
            self.vega_zps = vega_zps

        for band in self.model_filters:
            if band not in self.transmission_profiles.keys():
                if not os.path.exists(f'{code_dir}/{save_folder}/{band}.csv'):
                    print(f'Fetching {band} transmission profile.')
                    filter_name = self.band_codes[band]
                    filter_profile = SvoFps.get_transmission_data(filter_name)
                    wav = np.array(filter_profile["Wavelength"]) # Angstrom
                    trans = np.array(filter_profile["Transmission"])
                    out = np.vstack((wav, trans)).T
                    if not os.path.exists(f'{code_dir}/{save_folder}'):
                        os.makedirs(f'{code_dir}/{save_folder}')

                    np.savetxt(f'{code_dir}/{save_folder}/{band}.csv', out, delimiter=',', header='Wavelength (Angstrom), Transmission')
                else:
                    wav, trans = np.loadtxt(f'{code_dir}/{save_folder}/{band}.csv', delimiter=',', unpack=True)

                self.transmission_profiles[band] = (wav, trans)

    def model_parameter_ranges(self, model_version='sonora_bobcat'):
        return self.model_parameters[model_version]

    def model_file_extensions(self, model_version='sonora_bobcat'):
        model_ext = {'sonora_bobcat':'', 'sonora_cholla':'.spec', 'sonora_elf_owl':'.nc', 'sonora_diamondback':'.spec', 'low-z':'.txt', 'ATMO2020':'.txt'}

        return model_ext[model_version]

    def _latex_label(self, param):
        self.latex_labels = { 'temp': r'$T_{\rm eff}$', 'log_g': r'$\log g$', 'met': r'$\rm [Fe/H]$', 'kzz': r'$\rm K_{zz}$', 'co': r'$\rm C/O$', 'f':r'$\rm fsed', 'kzz_type': 'Kzz Model: ', 'model': 'Evo. Model: '}
        return self.latex_labels[param]

    def param_unit(self, param):
        self.units = {'temp': u.K, 'log_g': u.cm/u.s**2, 'met': u.dimensionless_unscaled, 'kzz': u.dimensionless_unscaled, 'co': u.dimensionless_unscaled, 'f':u.dimensionless_unscaled, 'kzz_type': u.dimensionless_unscaled}
        return self.units[param]

    def convert_templates(self, out_folder='sonora_model/', model_version='bobcat', overwrite=False):

        model_parameter_ext = self.model_file_extensions(model_version)

        # Count the number of models to convert with extension recurisvely in the file system
        resampled_files = glob.glob(f'{out_folder}/{model_version}/resampled/*_resample.dat', recursive=True)
        all_files = glob.glob(f'{self.library_path}/{model_version}/**/*{model_parameter_ext}', recursive=True)
        files_to_ignore = ['parameter']
        all_files = [file for file in all_files if not any([i in file for i in files_to_ignore])]

        all_files = [file for file in all_files if '_resample' not in file]
        all_files = [file for file in all_files if '.gz' not in file and '.tar' not in file and '.zip' not in file]
        all_files = [file for file in all_files if Path(file).is_file()]
        all_file_names = [file.split('/')[-1] for file in all_files]
        npermutations = len(all_files)

        print(f'Total resampled files: {len(resampled_files)}, Left to convert: {np.max([npermutations - len(resampled_files), 0])}')

        count = 0
        processed_files = []
        if os.path.exists(f'{out_folder}/{model_version}.param') and not overwrite:
            # count the number of models already converted
            with open(f'{out_folder}/{model_version}.param', 'r') as f:
                lines = f.readlines()
                count = len(lines)
                print(f'Already converted: {count} models.')
                processed_files = [i.strip() for i in lines]


        if count >= npermutations:
            return

        if model_version == 'sonora_elf_owl':
            # Working out folder name
            possible_files = os.listdir(f'{self.library_path}/{model_version}/')
            possible_files = [i for i in possible_files if os.path.isdir(f'{self.library_path}/{model_version}/{i}')]
            possible_files = [i for i in possible_files if 'resampled' not in i]
            options = [(float(i.split('_')[1]), float(i.split('_')[2])) for i in possible_files]

        if not os.path.exists(f'{out_folder}/{model_version}/resampled/'):
            os.makedirs(f'{out_folder}/{model_version}/resampled/')

        with open(f'{out_folder}/{model_version}.param', 'a') as f:
            for _, params in tqdm(iterate_model_parameters(self.model_parameters, model_version), total=npermutations, desc=f'Converting {model_version} models'):

                temp = params['temp']
                log_g = params['log_g']

                if model_version == 'sonora_bobcat':
                    m = params['met']
                    co = params['co']
                    mname = f'_m+{m}/spectra/' if m == '0.0' and co == '' else f'_m{m}'
                    mname = f'_m+{m}' if co != '' else mname

                    co_name = f'_co{co}' if co != '' else ''
                    co_folder_name = f'{co_name}_g1000nc' if co != '' else ''
                    m = f'+{m}' if co != '' and m == '0.0' else m
                    path = f'{self.library_path}/{model_version}/spectra{mname}{co_folder_name}/'

                    name = f'sp_t{temp}g{log_g}nc_m{m}{co_name}'
                    name_new = f'sp_t{temp}g{log_g}nc_m{m}{co_name}_resample.dat'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    #else:
                    #    print(f'/spectra{mname}{co_folder_name}/{name}')

                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'
                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            continue

                elif model_version == 'sonora_cholla':
                    kzz = params['kzz']
                    name = f'{temp}K_{log_g}g_logkzz{kzz}.spec'
                    path = f'{self.library_path}/{model_version}/spectra/spectra_files/'
                    name_new = f'{temp}K_{log_g}g_logkzz{kzz}_resample.dat'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'

                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            continue

                elif model_version == 'sonora_elf_owl':
                    m = params['met']
                    co = params['co']
                    kzz = params['kzz']

                    for i in options:
                        if temp >= i[0] and temp <= i[1]:
                            temp_low, temp_high = i
                            if temp_high> 550: # i hate inconsistent file naming
                                temp_high = int(temp_high)
                            break

                    path = f'{self.library_path}/{model_version}/output_{temp_low}_{temp_high}/'
                    name = f'spectra_logzz_{kzz}_teff_{float(temp)}_grav_{log_g}_mh_{m}_co_{float(co)}.nc'
                    name_new = f'spectra_logzz_{kzz}_teff_{float(temp)}_grav_{log_g}_mh_{m}_co_{float(co)}_resample.dat'
                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            continue

                elif model_version == 'sonora_diamondback':
                    m = params['met']
                    co = params['co']
                    fi = params['f']


                    name = f't{temp}g{log_g}{fi}_m{m}_co{co}.spec'
                    path = f'{self.library_path}/{model_version}/spec/spectra/'
                    name_new = f't{temp}g{log_g}{fi}_m{m}_co{co}_resample.dat'
                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            continue

                elif model_version == 'low-z':
                    m = params['met']
                    co = params['co']
                    kzz = params['kzz']
                    name = f'LOW_Z_BD_GRID_CLEAR_Teff_{float(temp)}_logg_{float(log_g)}_logZ_{m}_CtoO_{co}_logKzz_{kzz}_spec.txt'
                    path = f'{self.library_path}/{model_version}/models/models/'
                    name_new = name[:-4]+'_resample.dat'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'
                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, resample=False, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            print('no result')
                            continue
                        print(result)

                elif model_version == 'ATMO2020':
                    kzz_type = params['kzz_type']
                    if kzz_type == 'none':
                        folder = 'CEQ'
                    elif kzz_type == 'weak':
                        folder = 'NEQ_weak'
                    elif kzz_type == 'strong':
                        folder = 'NEQ_strong'

                    path = f'{self.library_path}/{model_version}/models/atmosphere_models/{folder}_spectra/'
                    name = f'spec_T{int(temp)}_lg{log_g}_{folder}.txt'
                    name_new = f'spec_T{int(temp)}_lg{log_g}_{folder}_resample.dat'

                    if f'{model_version}/resampled/{name_new}' in processed_files:
                        all_file_names.remove(name)
                        continue

                    new_path = f'{out_folder}/{model_version}/resampled/{name_new}'
                    if not Path(new_path).is_file() or overwrite:
                        result = self._resample_model(f'{path}/{name}', model_version=model_version, meta=params, new_path=new_path)
                        if result == 'no overlap':
                            all_file_names.remove(name)
                            continue
                        if not result:
                            continue

                else:
                    raise Exception(f'Unknown model version: {model_version}')

                if f'{model_version}/resampled/{name_new}' not in processed_files:
                    f.writelines(f'{model_version}/resampled/{name_new}\n')
                if name in all_file_names:
                    all_file_names.remove(name)
                count += 1

        if len(all_file_names) > 0:
            for file in all_file_names:
                if self.verbose:
                    print('Failed to convert:', file)

        if model_version not in ['ATMO2020', 'low-z']:
            assert len(all_file_names) == 0, f'Failed to convert {len(all_file_names)} models.'

    def clear_resampled_models(self, model_version='sonora_bobcat'):
        model_file_ext = self.model_file_extensions(model_version)
        resampled_files = glob.glob(f'{self.library_path}/{model_version}/**/*_resample.dat', recursive=True)
        for file in resampled_files:
            os.remove(file)

    def _resample_model(self, path, model_version, resample=True, meta={}, new_path=''):
        try:
            if model_version == 'sonora_bobcat':
                with(open(path, 'r')) as f:
                    table = Table.read(path, format='ascii', data_start=2, header_start=None, guess=False, fast_reader=False, names=['microns', 'Flux (erg/cm^2/s/Hz)'], units=[u.micron, u.erg/(u.cm**2*u.s*u.Hz)])
            elif model_version == 'sonora_cholla':
                table = Table.read(path, format='ascii', data_start=2, header_start=None, guess=False, delimiter=r'\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
            elif model_version == 'sonora_elf_owl':
                import xarray
                ds = xarray.load_dataset(path)
                wav = ds['wavelength'].data * u.micron
                flux = ds['flux'].data * u.erg/u.cm**2/u.s/u.cm # erg/cm^2/s/cm
                flux = flux.to(u.erg/u.cm**2/u.s/u.Hz, equivalencies=u.spectral_density(wav))
                wav = wav.to(u.micron)
                table = Table([wav, flux], names=['microns', 'Flux (erg/cm^2/s/Hz)'])
            elif model_version == 'sonora_diamondback':
                table = Table.read(path, format='ascii', data_start=3, header_start=None, guess=False, delimiter=r'\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
            elif model_version == 'low-z':
                table = Table.read(path, format='ascii', data_start=1, header_start=None, guess=False, delimiter=r'\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
            elif model_version == 'ATMO2020':
                table = Table.read(path, format='ascii', data_start=1, header_start=None, guess=False, delimiter=r'\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))

        except FileNotFoundError:
            #if self.verbose:
            #print(f'File not found: {path}')
            return False

        table['wav'] = table['microns'].to(u.AA)
        table.sort('wav')
        table_order = table['wav', 'Flux (erg/cm^2/s/Hz)']
        table_order = table_order[(table['wav'] > self.min_wav) & (table['wav'] < self.max_wav)]
        if len(table_order) == 0:
            if self.verbose:
                print('No overlap with filter range.')
            return 'no overlap'


        new_table = Table()

        if resample:
            if self.constant_R:
                new_disp_grid = self._generate_wav_sampling([table_order['wav'].max()], table_order['wav'].min()) * u.AA
            else:
                new_disp_grid = np.arange(table_order['wav'].min(), table_order['wav'].max(), self.resample_step) * u.AA

            fluxes = spectres.spectres(new_disp_grid.to(u.AA).value, table_order['wav'].to(u.AA).value, table_order['Flux (erg/cm^2/s/Hz)'].value, fill=np.nan, verbose=False)

            new_table['wav'] = new_disp_grid
            new_table['flux_nu'] = fluxes * u.erg/(u.cm**2*u.s*u.Hz)

        else:
            new_table['wav'] = table_order['wav']
            new_table['flux_nu'] = table_order['Flux (erg/cm^2/s/Hz)']

        #header = [i.replace(',', '') for i in header.split(' ') if i not in ['', '\n', ' ', '(MKS),']]
        #new_table.meta = {header[i+len(header)//2]:header[i] for i in range(len(header)//2)}

        if meta != {}:
            new_table.meta = meta

        ext = self.model_file_extensions(model_version)
        if path.endswith(ext):
            path = path[:-len(ext)]

        if new_path == '':
            new_path=path+'_resample.dat'

        new_table.write(new_path, format='ascii.ecsv', overwrite=True)
        return True

    def _generate_wav_sampling(self, max_wavs, min_wav=1):
        if type(self.R) not in [list, np.ndarray]:
            R = [self.R]
        else:
            R = self.R
        # Generate the desired wavelength sampling.
        x = [min_wav]

        for i in range(len(R)):
            if i == len(R)-1 or R[i] > R[i+1]:
                while x[-1] < max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/R[i]))

            else:
                while x[-1]*(1.+0.5/R[i]) < max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/R[i]))

        return np.array(x)

    def _fit_lsq(self, template_grid, filter_mask=None,  subset=None, sys_err=None, apply_extcorr=False):

        wave_lim = (self.min_wav, self.max_wav)

        if sys_err is None:
            sys_err = 0.0

        star_flux = np.array(template_grid).T
        self.NSTAR = star_flux.shape[1]

        fnu = self.fnu.copy()
        efnu = self.efnu.copy()

        fnu[~self.ok_data] = -99 #np.ma.masked_array(self.fnu, mask=~self.ok_data)
        efnu[~self.ok_data] = -99 #np.ma.masked_array(self.efnu, mask=~self.ok_data)

        # Least squares normalization of stellar templates
        if subset is None:
            _wht = 1/(efnu**2+(sys_err*fnu)**2)

            _wht /= self.zp**2
            _wht[(~self.ok_data) | (self.efnu <= 0)] = 0

        else:
            _wht = 1.0 / (
                efnu[subset,:]**2 + (sys_err * fnu[subset,:])**2
            )

            _wht /= self.zp**2
            _wht[(~self.ok_data[subset,:]) | (self.efnu[subset,:] <= 0)] = 0

        # if we want to use this, will need to provide grid just for subset templates bands
        #clip_filter = (self.pivot < wave_lim[0]) | (self.pivot > wave_lim[1])
        #clip_filter = np.full(self.nusefilt, False, dtype=bool)

        #if filter_mask is not None:
        #        clip_filter &= filter_mask

        #if filter_mask is not None:
        #    _wht[:, filter_mask] = 0

        if subset is None:
            _num = np.dot(fnu * self.zp * _wht, star_flux)
        else:
            _num = np.dot(fnu[subset,:] * self.zp * _wht, star_flux)

        #print(np.shape(_wht), np.shape(filter_mask), np.shape(template_grid), np.shape(star_flux), np.shape(_num))

        _den= np.dot(1*_wht, star_flux**2)
        _den[_den == 0] = 0
        star_tnorm = _num/_den


        # Chi-squared
        star_chi2 = np.zeros(star_tnorm.shape, dtype=np.float32)
        for i in tqdm(range(self.NSTAR), desc='Calculating chi2 for all templates...'):
            _m = star_tnorm[:,i:i+1]*star_flux[:,i]
            if subset is None:
                star_chi2[:,i] = np.sum(
                    (fnu * self.zp - _m)**2 * _wht
                , axis=1)
            else:
                star_chi2[:,i] = np.sum(
                    (fnu[subset,:] * self.zp - _m)**2 * _wht
                , axis=1)

        # Mask rows where all elements are NaN
        nan_rows = np.all(np.isnan(star_chi2), axis=1)
        star_min_ix = np.full(star_chi2.shape[0], -1, dtype = int)
        # Handle normally for valid rows
        # Compute nanargmin only for valid rows
        valid_rows = ~nan_rows
        star_min_ix[valid_rows] = np.nanargmin(star_chi2[valid_rows], axis=1)
        #star_min_ix = np.nanargmin(star_chi2, axis=1)
        star_min_chi2 = np.nanmin(star_chi2, axis=1)

        if subset is None:
            star_min_chinu = star_min_chi2 / (self.nusefilt - 1)
            '''star_gal_chi2 = (
                (self.fnu * self.zp - self.fmodel)**2 *_wht
            ).sum(axis=1)'''
        else:
            star_min_chinu = star_min_chi2 / (self.nusefilt[subset] - 1)
            ''' star_gal_chi2 = (
                (self.fnu[subset,:] * self.zp - self.fmodel[subset,:])**2 * _wht
            ).sum(axis=1)
            '''
        if subset is None:
            # Set attributes
            self.star_tnorm = star_tnorm
            self.star_chi2 = star_chi2
            self.star_min_ix = star_min_ix
            self.star_min_chi2 = star_min_chi2
            self.star_min_chinu = star_min_chinu
            #self.star_gal_chi2 = star_gal_chi2

        result = dict(
            subset = subset,
            star_tnorm = star_tnorm,
            star_chi2 = star_chi2,
            star_min_ix = star_min_ix,
            star_min_chi2 = star_min_chi2,
            star_min_chinu = star_min_chinu,
            #star_gal_chi2 = star_gal_chi2,
        )

        return result

    def _fit_lsq_mask(self, template_grid, filter_mask=None, subset=None, sys_err=None, apply_extcorr=False):

        wave_lim = (self.min_wav, self.max_wav)

        if sys_err is None:
            sys_err = 0.0

        star_flux = np.array(template_grid).T
        self.NSTAR = star_flux.shape[1]

        fnu = self.fnu.copy()
        efnu = self.efnu.copy()

        fnu[~self.ok_data] = -99
        efnu[~self.ok_data] = -99

        # Create a weight array
        if subset is None:
            _wht = 1/(efnu**2+(sys_err*fnu)**2)
            _wht /= self.zp**2
            _wht[(~self.ok_data) | (self.efnu <= 0)] = 0
        else:
            _wht = 1.0 / (
                efnu[subset,:]**2 + (sys_err * fnu[subset,:])**2
            )
            _wht /= self.zp**2
            _wht[(~self.ok_data[subset,:]) | (self.efnu[subset,:] <= 0)] = 0

        # Create a mask for template fluxes if provided
        template_mask = None
        if filter_mask is not None:
            if filter_mask.shape != template_grid.shape:
                raise ValueError("filter_mask must have the same shape as template_grid")

            # Transpose to match star_flux orientation
            template_mask = filter_mask.T

            # Apply template mask to weights
            if subset is None:
                # Broadcasting the mask to match _wht shape
                for i in range(star_flux.shape[1]):
                    _wht[:, template_mask[:, i] == 0] = 0
            else:
                # For subset, apply carefully to maintain dimensions
                for i in range(star_flux.shape[1]):
                    _wht[template_mask[:, i] == 0] = 0

        # Calculate normalization factors
        if subset is None:
            _num = np.dot(fnu * self.zp * _wht, star_flux)
        else:
            _num = np.dot(fnu[subset,:] * self.zp * _wht, star_flux)

        _den = np.dot(1*_wht, star_flux**2)

        # Avoid division by zero
        _den[_den == 0] = np.nan  # Use NaN instead of 0 to properly identify invalid normalizations
        star_tnorm = _num/_den

        #print(np.shape(_wht), np.shape(filter_mask), np.shape(template_grid), np.shape(star_flux), np.shape(_num))

        # Chi-squared calculation
        star_chi2 = np.zeros(star_tnorm.shape, dtype=np.float32)
        for i in tqdm(range(self.NSTAR), desc='Calculating chi2 for all templates...'):
            if np.isnan(star_tnorm[:, i]).all():
                star_chi2[:, i] = np.nan
                continue

            _m = star_tnorm[:,i:i+1]*star_flux[:,i]
            if subset is None:
                star_chi2[:,i] = np.sum(
                    (fnu * self.zp - _m)**2 * _wht
                , axis=1)
            else:
                star_chi2[:,i] = np.sum(
                    (fnu[subset,:] * self.zp - _m)**2 * _wht
                , axis=1)

        # Handle NaN rows
        nan_rows = np.all(np.isnan(star_chi2), axis=1)
        star_min_ix = np.full(star_chi2.shape[0], -1, dtype=int)

        # Compute nanargmin only for valid rows
        valid_rows = ~nan_rows
        if np.any(valid_rows):  # Check if there are any valid rows
            star_min_ix[valid_rows] = np.nanargmin(star_chi2[valid_rows], axis=1)
            star_min_chi2 = np.nanmin(star_chi2, axis=1)
        else:
            # If all rows are invalid, set everything to NaN
            star_min_chi2 = np.full(star_chi2.shape[0], np.nan)

        if subset is None:
            # Avoid division by zero in chi-squared per degree of freedom
            dof = self.nusefilt - 1
            # Correct DOF for bands which are masked for this template
            if filter_mask is not None:
                dof -= np.sum(filter_mask[star_min_ix, :] == 0, axis=1)


            dof = np.maximum(dof, 1)  # Ensure we don't divide by zero
            star_min_chinu = star_min_chi2 / dof
        else:
            dof = self.nusefilt[subset] - 1

            if filter_mask is not None:
                dof -= np.sum(filter_mask[star_min_ix, :] == 0, axis=1)

            dof = np.maximum(dof, 1)  # Ensure we don't divide by zero
            star_min_chinu = star_min_chi2 / dof

        if subset is None:
            # Set attributes
            self.star_tnorm = star_tnorm
            self.star_chi2 = star_chi2
            self.star_min_ix = star_min_ix
            self.star_min_chi2 = star_min_chi2
            self.star_min_chinu = star_min_chinu

        result = dict(
            subset = subset,
            star_tnorm = star_tnorm,
            star_chi2 = star_chi2,
            star_min_ix = star_min_ix,
            star_min_chi2 = star_min_chi2,
            star_min_chinu = star_min_chinu,
        )

        return result

    def _catalogue_mask_bands(self, bands, library):
        ''' 

        Makes a 2D mask for the template grid to avoid fitting bands which fully or partially fall outside the wavelength range of the library.
        '''
        grid = []
        for pos, name in enumerate(self.template_names[library]):
            min_wav, max_wav = self.template_ranges[library][pos]
            fitted_bands = np.array([True if self.filter_wavs[band] >= min_wav and self.filter_wavs[band] <= max_wav else False for band in bands])
            grid.append(fitted_bands)

        return np.array(grid, dtype=bool)

    def fit_catalog(self,
                photometry_function: Callable = None,
                bands: List[str] = 'internal',
                photometry_function_kwargs: dict = {},
                libraries_to_fit: Union[str, List[str]] = 'all',
                sys_err=None,
                filter_mask=None,
                subset=None,
                fnu=None,
                efnu=None,
                catalogue_ids=None,
                dump_fit_results=False,
                dump_fit_results_path='fit_results.pkl',
                outside_wav_range_behaviour='clip'):
        '''
        Photometry function should be a function that returns the fluxes and flux errors to be fit. Or directly provide the fluxes and errors.

        Parameters
        ----------
        photometry_function : Callable
            Function that returns the fluxes and flux errors to be fit.
        bands : List[str]
            List of bands to fit.
        photometry_function_kwargs : dict, optional
            Keyword arguments for the photometry function. The default is {}.
        libraries_to_fit : Union[str, List[str]], optional
            Libraries to fit. The default is 'all'.
        sys_err : float, optional
            Systematic error to include in the fit. The default is None.
        filter_mask : np.ndarray, optional
            Mask to apply to the filters. The default is None.
        subset : np.ndarray, optional
            Subset of templates to fit. The default is None.
        fnu : np.ndarray, optional
            Fluxes to fit if not using photometry_function. The default is None.
        efnu : np.ndarray, optional
            Errors to fit if not using photometry_function. The default is None.
        catalogue_ids : List, optional
            List of catalogue ids. The default is None.
        dump_fit_results : bool, optional
            Whether to dump the fit results to a pickle file. The default is False.
        dump_fit_results_path : str, optional
            Path to dump the fit results. The default is 'fit_results.pkl'.
        outside_wav_range_behaviour : str, optional
            Behaviour for when the wavelength range of the bands to fit is outside the wavelength range of the library. The options
            are either 'clip' to exlude bands or 'subset' to only fit the bands that are within the library wavelength range. The default is 'clip'.


        '''

        assert photometry_function is not None or (fnu is not None and efnu is not None), 'Provide either a photometry function or fluxes and errors.'

        if bands == 'internal':
            bands = self.model_filters


        if photometry_function is not None:
            fnu, efnu = photometry_function(**photometry_function_kwargs)

        # Helps with fitting one object catalogues
        fnu = np.atleast_2d(fnu)
        efnu = np.atleast_2d(efnu)

        assert len(fnu) == len(efnu), 'Flux and error arrays must be the same length.'
        assert len(fnu[0]) == len(bands), f'Flux and error arrays must have the same number of bands, got {len(fnu[0])} and {len(bands)}.'
        assert type(fnu) is u.Quantity, 'Fluxes must be astropy Quantities.'
        assert type(efnu) is u.Quantity, 'Errors must be astropy Quantities.'

        self.catalogue_ids = catalogue_ids
        if catalogue_ids is not None:
            assert len(catalogue_ids) == len(fnu), 'Catalogue IDs must be the same length as the fluxes.'

        self.fnu = fnu.to(u.nJy).value
        self.efnu = efnu.to(u.nJy).value


        # Get template grid here.
        if libraries_to_fit == 'all':
            libraries_to_fit = self.libraries

        # Make mask for columns in self.model_filters that aren't in bands

        split_model_filters = [i.split('.')[-1] for i in self.model_filters]

        fitted_bands = []
        phot_bands = []
        band_idx = []
        band_compar_dict = {}
        for band in bands:
            if band not in self.model_filters:
                # check all filters in split_model_filters are unique
                #if len(split_model_filters) != len(set(split_model_filters)):
                #    duplicates = [item for item, count in collections.Counter(split_model_filters).items() if count > 1]
                #    raise Exception(f'Didn\'t recognize {band} and the attempt to compare to filters without instrument names resulted in duplicate {duplicates}. Please provide the full filter name. E.g. ACS_WFC.F814W')
                if band in split_model_filters:
                    # Check if band is in split_model_filters more than once
                    if split_model_filters.count(band) > 1:
                        raise Exception(f'Band {band} is in internal filter dictionary more than once. Please provide the full filter name. E.g. ACS_WFC.F814W')
                    match_idx = split_model_filters.index(band)
                    full_band = self.model_filters[match_idx]
                    print(f'Warning! Assuming {band} is the same as {full_band}')
                    self.filter_wavs[band] = self.filter_wavs[full_band]
                    self.filter_ranges[band] = self.filter_ranges[full_band]
                    self.filter_instruments[band] = self.filter_instruments[full_band]
                    band_compar_dict[band] = full_band
                    phot_bands.append(full_band)
                    fitted_bands.append(full_band)
                    band_idx.append(match_idx)
                else:

                    print(f'Warning! Band {band} not in model_filters. Removing from bands to fit.')
                    bands.remove(band)
            else:
                phot_bands.append(band)
                fitted_bands.append(band)
                band_compar_dict[band] = band
                #print(f'Found {band}')
                band_idx.append(self.model_filters.index(band))

        '''
        min_wav, max_wav = self.wavelength_range_of_bands(fitted_bands)
        check_band_ranges = False

        extreme_min_wav, extreme_max_wav = 0 * u.um, np.inf * u.um
        for library in libraries_to_fit:
            library_min_wav, library_max_wav = self.wavelength_range_of_library(library)
            library_extreme_min_wav, library_extreme_max_wav = self.extreme_wavelength_range_of_library(library)

            print(f'Library {library} wavelength range: {library_min_wav.to(u.um)} - {library_max_wav.to(u.um)}')
            if library_min_wav > min_wav:
                print(f'Warning! Minimum wavelength of library {library} is greater than minimum wavelength of bands to fit. {library_min_wav.to(u.um):.2f} > {min_wav.to(u.um):.2f}')
                if min_wav > library_extreme_min_wav:
                    print(f'You can extend this range to {library_extreme_min_wav.to(u.um)} to include the minimum wavelength of the library {library} by recompting the photometry grid.') 
                check_band_ranges = True

            if library_max_wav < max_wav:
                print(f'Warning! Maximum wavelength of library {library} is less than maximum wavelength of bands to fit. {library_max_wav.to(u.um):.2f} < {max_wav.to(u.um):.2f}')
                if max_wav < library_extreme_max_wav:
                    print(f'You can extend this range to {library_extreme_max_wav.to(u.um):.2f} to include the maximum wavelength of the library {library} by recompting the photometry grid.')
                check_band_ranges = True

            extreme_min_wav = max(extreme_min_wav, library_extreme_min_wav)
            extreme_max_wav = min(extreme_max_wav, library_extreme_max_wav)

        if check_band_ranges:
            if outside_wav_range_behaviour == 'clip':
                print('Clipping bands to fit to the wavelength range of the library.')
                fitted_bands = [band for band in fitted_bands if self.filter_wavs[band] >= extreme_min_wav and self.filter_wavs[band] <= extreme_max_wav]

            elif outside_wav_range_behaviour == 'subset':
                print('Subsetting templates to fit to the wavelength range of the bands.')
                libraries_to_fit = [library for library in libraries_to_fit if self.wavelength_range_of_library(library)[0] < min_wav and self.wavelength_range_of_library(library)[1] > max_wav]

                if len(libraries_to_fit) == 0:
                    raise Exception('No libraries to fit. Please provide bands that are within the wavelength range of the library or change the value of outside_wav_range_behaviour.')
            else:
                raise Exception('Unknown value for outside_wav_range_behaviour. Please provide either "clip" or "subset".')
        '''

        grid = self._build_combined_template_grid(libraries_to_fit)

        print(f'Fitting with {", ".join(libraries_to_fit)} libraries with {np.shape(grid)[1]} templates.')

        mask = np.array([i in fitted_bands for i in self.model_filters])
        # Check actual bands are in the same order as self.model_filters
        self.bands_to_fit = fitted_bands
        self.bands_to_fit = [
            band for band in self.model_filters if band in fitted_bands
        ]
        self.mask = mask
        self.reduced_template_grid = grid[self.mask, :].T

        idxs = np.array([i for i, band in enumerate(phot_bands) if band in self.bands_to_fit])

        phot_to_model_order = np.array(
            [phot_bands.index(band) for band in self.bands_to_fit]
        )
        idxs = phot_to_model_order

        total_filter_mask = np.ones_like(self.reduced_template_grid, dtype=bool)
        for library in libraries_to_fit:
            filter_mask = self._catalogue_mask_bands(self.bands_to_fit, library)
            total_filter_mask[self.idx_ranges[library][0]:self.idx_ranges[library][1], :] = filter_mask

        self.total_filter_mask = total_filter_mask

        print(f'Fitting {len(fitted_bands)} bands: {fitted_bands}')

        # Generate ok_data to mask nan fluxes and nan errors

        self.NFILT = len(fitted_bands)
        self.fnu = self.fnu[:, idxs]
        self.efnu = self.efnu[:, idxs]
        ok_data = np.isfinite(self.fnu) & np.isfinite(self.efnu) & (self.efnu > 0)
        self.zp = np.ones_like(self.fnu)
        self.ok_data = ok_data
        self.nusefilt = self.ok_data.sum(axis=1)

        assert len(self.bands_to_fit) == self.NFILT, f'Number of bands to fit does not match number of bands in fluxes: {len(self.bands_to_fit)} != {self.NFILT}'

        # Check that all bands are in the model_filters

        fit_results = self._fit_lsq_mask(self.reduced_template_grid, filter_mask=total_filter_mask, subset=subset, sys_err=sys_err)

        if dump_fit_results:
            keys_to_dump = ['fnu', 'efnu', 'zp', 'ok_data', 'nusefilt', 'bands_to_fit', 'catalogue_ids', 'mask']
            dump_dict = {key: getattr(self, key) for key in keys_to_dump}
            dump_dict.update(fit_results)

            if '/' in dump_fit_results_path:
                path = dump_fit_results_path.split('/')
                path = '/'.join(path[:-1])
                if not os.path.exists(path):
                    os.makedirs(path)

            if not dump_fit_results_path.endswith('.pkl'):
                dump_fit_results_path = f'{dump_fit_results_path}.pkl'

            with open(dump_fit_results_path, 'wb') as f:
                import pickle
                pickle.dump(dump_dict, f, protocol=5) # Efficient binary protocol for large files

        return fit_results

    def get_catalog_physical_params(self):
        assert hasattr(self, 'star_tnorm'), 'Fit the catalogue first.'
        assert hasattr(self, 'star_min_ix'), 'Fit the catalogue first.'

        distances = np.zeros(len(self.star_min_ix))
        masses = np.zeros(len(self.star_min_ix))
        radii = np.zeros(len(self.star_min_ix))

        for i in range(len(self.star_min_ix)):
            norm = self.star_tnorm[i, self.star_min_ix[i]]
            results = self.get_physical_params(self.star_min_ix[i], norm)
            if results is not None:
                distance = results['distance']
                mass = results['mass']
                radius = results['radius']

            if mass is not None:
                masses[i] = mass.to(u.M_sun).value
            if radius is not None:
                radii[i] = radius.to(u.R_sun).value
            if distance is not None:
                distances[i] = distance.to(u.pc).value

        distances = distances * u.pc
        masses = masses * u.M_sun
        radii = radii * u.R_sun

        return distances, masses, radii

    def load_results_from_pickle(self, path):
        import pickle
        with open(path, 'rb') as f:
            results = pickle.load(f)

        for key in results.keys():
            if key == 'mask':
                self.reduce_template_grid = self.combined_template_grid[results[key], :].T

            setattr(self, key, results[key])

    def plot_fit(self, idx=None, cat_id=None, wav_unit=u.micron, flux_unit=u.nJy, override_template_ix=None, test_scale=1):

        assert idx is not None or cat_id is not None, 'Provide either an index or a catalogue id. (if catalogue IDs were provided during fitting'

        if cat_id is not None and idx is not None:
            raise Exception('Provide either an index or a catalogue id, not both.')

        if cat_id is not None:
            idx = np.where(self.catalogue_ids == cat_id)[0][0]
            pname = f'id: {cat_id} (idx: {idx})'
        else:
            pname = f'idx: {idx}'
            if self.catalogue_ids is not None:
                pname = f'{pname} (id: {self.catalogue_ids[idx]})'

        wavs = np.array([self.filter_wavs[band].to(wav_unit).value for band in self.bands_to_fit])
        flux = self.fnu[idx] * u.nJy
        flux_err = self.efnu[idx] * u.nJy

        # Make SED plot and normalized residual plot
        fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=200)

        if flux_unit != u.ABmag:
            err = flux_err.to(flux_unit, equivalencies=u.spectral_density(wavs*wav_unit)).value
        else:
            flux = flux.to(u.Jy)
            flux_err = flux_err.to(u.Jy)

            err_low = np.abs(2.5*np.log10(1 - flux_err/flux))
            err_high = np.abs(2.5*np.log10(1 + flux_err/flux))
            err = np.array([err_high, err_low])
            ax[0].invert_yaxis()

        plot_flux = flux.to(flux_unit, equivalencies=u.spectral_density(wavs*wav_unit)).value

        ax[0].errorbar(wavs, plot_flux, yerr=err, fmt='o', label='Phot.',
                        color='crimson', markeredgecolor='k', zorder=10)

        ax[0].set_xlim(ax[0].get_xlim())

        if flux_unit == u.ABmag:

            # plot any negative fluxes as 3 sigma upper limits
            for i, f in enumerate(flux):
                if f < 0:
                    from matplotlib.patches import FancyArrowPatch
                    error_3sigma = 3*flux_err[i]
                    error_3sigma = error_3sigma.to(flux_unit, equivalencies=u.spectral_density(wavs[i]*wav_unit)).value
                    length = abs(ax[0].get_ylim()[1] - ax[0].get_ylim()[0])*0.15
                    ax[0].add_patch(FancyArrowPatch((wavs[i], error_3sigma), (wavs[i], error_3sigma+length), facecolor='crimson', zorder=10, edgecolor='k', arrowstyle='-|>, scaleA=3', mutation_scale=8))

        if override_template_ix is not None:
            best_ix = override_template_ix
        else:
            best_ix = self.star_min_ix[idx]

        if best_ix != -1:
            model_phot = self.star_tnorm[idx, best_ix] * self.reduced_template_grid[best_ix] * u.nJy
            unmasked_bands = self.total_filter_mask[best_ix, :]
            plot_model_phot = model_phot.to(flux_unit, equivalencies=u.spectral_density(wavs*wav_unit)).value

            ax[0].scatter(wavs[unmasked_bands], plot_model_phot[unmasked_bands], label='Best Fit Phot.', facecolor='navy', edgecolor='white', zorder=10)
            ax[1].scatter(wavs[unmasked_bands], ((flux - model_phot) / flux_err)[unmasked_bands], facecolor='navy', edgecolor='white', zorder=10)

            ax[0].scatter(wavs[~unmasked_bands], plot_model_phot[~unmasked_bands], label='Masked Phot.', facecolor='navy', edgecolor='red', zorder=10, alpha=0.5)

            library, name = self.get_template_name(best_ix)

            self.plot_best_template(best_ix, idx, ax=ax[0], color='navy', wav_unit=wav_unit, flux_unit=flux_unit, linestyle='solid', lw=1)
            params = self._extract_model_meta(library, name)
            latex_labels = [self._latex_label(param) for param in params.keys()]

            #chi2 = np.nansum(((flux[unmasked_bands] - model_phot[unmasked_bands]) / flux_err[unmasked_bands])**2)
            results = self.get_physical_params(best_ix, self.star_tnorm[idx, best_ix])

            if results is not None:
                distance = results['distance']
                mass = results['mass']
                radius = results['radius']

            info_box = '\n'.join([f'{latex_labels[i]}: {params[param]}{self.param_unit(param):latex}' for i, param in enumerate(params.keys())])
            lower_info_box = rf'$\chi^2$: {self.star_min_chi2[idx]:.2f}\\n$\chi^2_\nu$: {self.star_min_chinu[idx]:.2f}\n'
            info_box = f'{pname}\nBest Fit: {library.replace("_", " ").title()}\n{info_box}\n{lower_info_box}'
            if results is not None:
                if library in self.parameter_overrides.keys():
                    grid = f'\nPhysical Parameters\nGrid: {str(self.parameter_overrides[library]["model"]).replace("_", " ").title()}\n'

                    if 'overrides' in results['info'].keys():
                        srtring = '\n'.join([f'{self._latex_label(key)}: {results["info"]["overrides"][key]}' for key in results["info"]["overrides"].keys() if key not in ['model']])
                        grid = f'{grid}{srtring}\n'

                else:
                    grid = ''

                lowest_info_box=f'{grid}Mass: {mass.to(u.Msun).to_string(format="latex", precision=3)}\nRadius: {radius.to_string(format="latex", precision=3)}\nDistance: {distance.to(u.kpc).to_string(format="latex", precision=3)}'
                # check other keys in results
                if len(results.keys()) > 3:
                    for key in results.keys():
                        if key not in ['mass', 'radius', 'distance', 'info']:
                            lowest_info_box = f'{lowest_info_box}\n{key}: {results[key].to_string(format="latex", precision=3)}'
                info_box = f'{info_box}\n{lowest_info_box}'
                self.plot_model_photometry(best_ix, ax=ax[0], norm = self.star_tnorm[idx, best_ix], color='navy', wav_unit=wav_unit, flux_unit=flux_unit, test_scale=test_scale, label = 'Distance Phot.')

            ax[0].text(1.02, 0.98, info_box, transform=ax[0].transAxes, fontsize=8, verticalalignment='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')], bbox=dict(facecolor='w', alpha=0.5, edgecolor='black', boxstyle='square,pad=0.5'))
            ax[1].vlines(wavs[unmasked_bands], ((flux - model_phot) / flux_err)[unmasked_bands], 0, color='k', alpha=0.5, linestyle='dotted')
            ax[1].set_ylim(np.nanmin(((flux - model_phot) / flux_err)[unmasked_bands]) - 0.2, np.nanmax(((flux - model_phot)/flux_err)[unmasked_bands])+0.2)
        else:
            print(f"No best fit found for {idx=}!")

        ax[0].set_ylabel(f'Flux Density ({flux_unit:latex_inline})')
        ax[1].set_xlabel(f'Wavelength ({wav_unit:latex_inline})')
        ax[1].set_ylabel('Residuals')
        fig.subplots_adjust(hspace=0)
        ax[0].set_xlim(ax[0].get_xlim())
        ax[0].set_ylim(ax[0].get_ylim())
        if flux_unit == u.ABmag:
            ax[0].set_ylim(32, ax[0].get_ylim()[1])
        ax[1].hlines(0, wavs[0], wavs[-1], linestyle='--', color='k')
        ax[0].legend()

        return fig, ax

    def make_cat(
        self,
        save_path: Optional[str] = None,
        meta_names: Optional[List[str]] = model_param_names,
    ):
        best_template_libraries_names = np.array([self.get_template_name(i) for i in self.star_min_ix])
        best_library_names = best_template_libraries_names[:, 0]
        best_template_names = best_template_libraries_names[:, 1]
        if meta_names is not None:
            best_model_params = np.array([self.get_template_parameters(i) for i in self.star_min_ix])
            best_template_meta = {
                meta_name: np.array(
                    [
                        meta[meta_name] if meta_name in meta.keys() else np.nan
                        for meta in best_model_params
                    ]
                ).astype(model_param_dtypes[meta_name])
                for meta_name in meta_names
            }
            meta_dtypes = [model_param_dtypes[name] for name in meta_names]
        else:
            best_template_meta = {}
            meta_dtypes = []

        norm = np.array([self.star_tnorm[idx, best_ix] if best_ix != -1
            else -1 for idx, best_ix in enumerate(self.star_min_ix)])
        phot = np.array([norm[idx] * self.reduced_template_grid[best_ix]
            if best_ix != -1 else np.full(len(self.bands_to_fit), -1)
            for idx, best_ix in enumerate(self.star_min_ix)])
        assert phot.shape[1] == len(self.bands_to_fit)
        phot_data = {f"{band}_nJy": phot[:, i] for i, band in enumerate(self.bands_to_fit)}

        tab = Table(
            data = {
                "best_template": best_template_names,
                "best_library": best_library_names,
                "chi2": self.star_min_chi2,
                "red_chi2": self.star_min_chinu,
                "template_norm": norm,
                **best_template_meta,
                **phot_data,
            },
            dtype = [str, str, float, float, float] + meta_dtypes + list(np.full(phot.shape[1], float)),
        )
        meta = {
            "scaling_factor": self.default_scaling_factor,
            "library_path": self.library_path,
        }
        tab.meta = meta
        if save_path is not None:
            tab.write(save_path, overwrite = True)
        return tab

    def make_best_fit_SEDs(
        self,
        save_path: str,
        wav_unit: u.Unit = u.micron,
        flux_unit: u.Unit = u.nJy,
        IDs: Optional[List[int]] = None,
    ):
        # save as .h5
        if not save_path[-3:] == ".h5":
            save_path = f"{'.'.join(save_path.split('.')[:-1])}.h5"

        if not os.path.exists(save_path):

            if IDs is not None:
                assert len(IDs) == len(self.star_min_ix), \
                    "IDs must be the same length as the number of objects."
                IDs = np.array(IDs).astype(int)
            else:
                IDs = np.arange(len(self.star_min_ix)).astype(int)

            libraries = np.array([self.get_template_name(best_ix)[0]
                if not best_ix == -1 else "" for best_ix in self.star_min_ix])
            valid_library = libraries != ""
            unique_libraries = np.unique(libraries[valid_library])
            libraries_IDs = {library: [] for library in unique_libraries}
            libraries_SEDs = deepcopy(libraries_IDs)
            for idx, library in tqdm(
                enumerate(libraries),
                desc = f"Loading best fit SEDs for {repr(self)}",
                total = len(libraries),
            ):
                if library != "":
                    libraries_IDs[library].extend([IDs[idx]])
                    libraries_SEDs[library].extend(
                        [self.load_SED(self.star_min_ix[idx], idx, wav_unit = wav_unit, flux_unit = flux_unit)]
                    )

            with h5.File(save_path, "w") as f:

                for library in unique_libraries:
                    # ensure the length of arrays are all the same, if not zero pad from the start
                    SED_lengths = np.array([len(SED[0]) for SED in libraries_SEDs[library]])
                    while not all(SED_lengths[0] == SED_length for SED_length in SED_lengths):
                        # zero pad the shorter arrays
                        longest_length_indices = np.where(SED_lengths == max(SED_lengths))[0]
                        for i, (SED, length) in enumerate(zip(libraries_SEDs[library], SED_lengths)):
                            if i not in longest_length_indices:
                                # TODO: ensure mis-matching wavelengths occurs at the start
                                new_SED_wavs = libraries_SEDs[library][longest_length_indices[0]][0]
                                new_SED_fluxes = np.concatenate([np.zeros(SED_lengths[longest_length_indices[0]] - length), SED[1]])
                                # zero pad the shorter array starting from the start
                                libraries_SEDs[library][i] = (new_SED_wavs, new_SED_fluxes)
                        SED_lengths = np.array([len(SED[0]) for SED in libraries_SEDs[library]])
                        if all(SED_lengths[0] == SED_length for SED_length in SED_lengths):
                            break

                    # ensure all wavelengths are the same
                    consistent_wavelengths = all(
                        all(
                            np.array(libraries_SEDs[library][0][0].to(wav_unit).value)
                            == np.array(library_SEDs[0].to(wav_unit).value)
                        )
                        for library_SEDs in libraries_SEDs[library]
                    )
                    assert consistent_wavelengths
                    consistent_flux_lengths = all(
                        len(libraries_SEDs[library][0][1]) == len(library_SEDs[1])
                        for library_SEDs in libraries_SEDs[library]
                    )

                    # ensure the length of the fluxes is the same
                    assert consistent_flux_lengths

                    h5_lib = f.create_group(library)
                    h5_lib.create_dataset(
                        "IDs",
                        data = np.array(libraries_IDs[library]).astype(int),
                        compression = "gzip",
                        dtype = int,
                    )
                    h5_lib.create_dataset(
                        "wavs",
                        data = np.array(libraries_SEDs[library][0][0]).astype(np.float32),
                        compression = "gzip",
                        dtype = np.float32,
                    )
                    h5_lib.create_dataset(
                        "fluxes",
                        data = np.array(libraries_SEDs[library])[:, 1, :].astype(np.float32),
                        compression = "gzip",
                        dtype = np.float32,
                    )
                    h5_lib.attrs["wav_unit"] = wav_unit.to_string()
                    h5_lib.attrs["flux_unit"] = flux_unit.to_string()
                f.close()
        else:
            print(f"{save_path} already exists. Skipping.")

    def get_template_name(self, model_idx):
        if model_idx == -1:
            return "", ""
        else:
            for library in self.libraries:
                if model_idx >= self.idx_ranges[library][0] and model_idx < self.idx_ranges[library][1]:
                    return library, self.template_names[library][model_idx - self.idx_ranges[library][0]]

    def load_SED(
        self,
        model_idx,
        input_idx,
        wav_unit = u.micron,
        flux_unit = u.nJy,
    ):
        library, name = self.get_template_name(model_idx)

        path = f'{self.library_path}/{library}/resampled/{name}'

        best_fit_coefficients = self.star_tnorm[input_idx, model_idx]

        table = Table.read(path, format='ascii.ecsv', delimiter=' ', names=['wav', 'flux_nu'], units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        table['flux_njy'] = best_fit_coefficients * table['flux_nu'].to(u.nJy) * self.scaling_factors[library]

        return table["wav"].to(wav_unit), table['flux_njy'].to(flux_unit)


    def plot_best_template(self, model_idx, input_idx, ax=None, wav_unit=u.micron, flux_unit=u.nJy, **kwargs):

        wav, flux = self.load_SED(model_idx, input_idx, wav_unit=wav_unit, flux_unit=flux_unit)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
            ax.set_xlabel(f'Wavelength ({wav_unit:latex_inline})')
            ax.set_ylabel(f'Flux Density ({flux_unit:latex_inline})')

        ax.plot(wav, flux, **kwargs)

    def _get_model_path_from_lib_name(self, library, name):
        return f'{self.library_path}/{library}/resampled/{name}'

    def get_template_parameters(self, model_idx):
        if model_idx == -1:
            return {}
        else:
            for library in self.libraries:
                if model_idx >= self.idx_ranges[library][0] and model_idx < self.idx_ranges[library][1]:
                    return {key: values[model_idx - self.idx_ranges[library][0]] for key, values in self.template_parameters[library].items()}

    def _extract_model_meta(self, library, name):
        path = self._get_model_path_from_lib_name(library, name)
        table = Table.read(path, format='ascii.ecsv', delimiter=' ', names=['wav', 'flux_nu'], units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        meta = table.meta
        return meta

    def color_color(self, x, y, libraries='all', unit=u.ABmag, filter_by=None,
                    show_fitted_galaxies=False, color_by=None, **kwargs):

        if libraries == 'all':
            libraries = self.libraries

        grid = self._build_combined_template_grid(libraries)

        # convert to AB mag

        if unit == u.ABmag:
            grid_converted = -2.5*np.log10(grid * 1e-9) + 8.9
        else:
            print('Warning: unit not recognized. Fluxes are arbitrary.')
            grid_converted = grid

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

        # Create dictionary of views into filter_data with filter names as keys
        filter_idx = {band.split('.')[0]: i for i, band in enumerate(self.model_filters)}
        filter_idx = {band:i for i, band in enumerate(self.model_filters)}

        filter_data = {band: grid_converted[i] for band, i in filter_idx.items()}

        parser = FilterArithmeticParser()

        x_data = parser.parse_and_evaluate(x, filter_data)
        y_data = parser.parse_and_evaluate(y, filter_data)

        if filter_by is not None:
            filter_mask = parser.parse_and_evaluate(filter_by, filter_data).astype(bool)
            x_data = x_data[filter_mask]
            y_data = y_data[filter_mask]
            ax.set_title(f'Filtered by: {filter_by}')
        else:
            filter_mask = np.ones_like(x_data, dtype=bool)

        if color_by is not None:
            if color_by == 'libraries':
                ncolors = len(libraries)
                idxs = self.idx_ranges
                colors = plt.cm.viridis(np.linspace(0, 1, ncolors))
                for i, library in enumerate(libraries):
                    idx_range = range(idxs[library][0], idxs[library][1])
                    ax.scatter(x_data[idx_range], y_data[idx_range], color=colors[i], label=library.replace('_', ' ').title(), **kwargs)
            elif color_by in model_param_dtypes.keys():
                # Color by a model parameter. Need to assemble the parameter data for all libraries
                param_data = []
                for library in libraries:
                    param_data.append(self.template_parameters[library][color_by])
                param_data = np.concatenate(param_data)
                sc = ax.scatter(x_data, y_data, c=param_data[filter_mask], cmap='viridis', **kwargs)
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(self._latex_label(color_by))
            else:
                raise Exception(f'color_by {color_by} not recognized.')
        else:
            ax.scatter(x_data, y_data, **kwargs)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        libraries_string = '\n'.join(libraries)

        if show_fitted_galaxies:
            if getattr(self, 'fnu', None) is None or getattr(self, 'efnu', None) is None or getattr(self, 'bands_to_fit', None) is None:
                raise Exception('No fitted galaxies to plot.')

            # need dictionary matching self.bands_to_fit and self.fnu
            band_data = self.fnu.T
            band_err = self.efnu.T

            band_data = {band: band_data[i] for i, band in enumerate(self.bands_to_fit)}
            band_err = {band: band_err[i] for i, band in enumerate(self.bands_to_fit)}

            if unit == u.ABmag:
                band_data = {band: -2.5*np.log10(band_data[band] * 1e-9) + 8.9 for band in band_data}
                band_err = {band: -2.5*np.log10(band_err[band] * 1e-9) + 8.9 for band in band_err}

            x_data = parser.parse_and_evaluate(x, band_data)
            y_data = parser.parse_and_evaluate(y, band_data)
        if color_by is not None and color_by != 'libraries':
            ax.text(0.05, 0.95, f'Libraries:\n{libraries_string}', transform=ax.transAxes, fontsize=8, verticalalignment='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])
        else:
            ax.legend()
        return fig, ax

    def filter_template_grid_by_parameter(self, parameter, value, exclude_unparametrized=True, combine='and'):
        '''
        TODO: FINISH THIS

        Filter the template grid by a parameter value. E.g. filter by temperature, log g, etc.

        Parameters
        ----------
        parameter : str or list of parameters - if allowing multiple parameters to be filtered by - then provide a list of parameters and values of the same length
            Parameter to filter by.
        value : float, string, list of allowed values or tuple range. - specific strings with special behaviour are 'T', 'L' and 'Y' which
        refer to temperature ranges for T dwarfs, L dwarfs and Y dwarfs respectively.
            Value to filter by.
        exclude_unparametrized : bool, optional
            Exclude templates that do not have the parameter. The default is True.
        combine: str, optional
            If filtering by multiple parameters, how to combine the filters. Options are 'and' and 'or'. The default is 'and'.


        Returns
        -------
        filtered_grid : np.ndarray
            Filtered template grid.
        '''

        param_aliases = {'temp': ['t', 'teff', 'temperature', 't_eff', 'temp'],
                        'log_g':['g', 'logg', 'gravity', 'log_g', 'log_g'],
                        'met':['Z', 'metallicity', 'zmet', 'met'],
                        'co':['c/o', 'co', 'c_o', 'co_ratio'],
                        'f':['f', 'fparam', 'f_param']}


        if type(parameter) is not list:
            parameter = [parameter]
            value = [value]

        assert len(parameter) == len(value), 'Length of parameters and values must be the same.'

        global_mask = np.ones(len(self.combined_template_grid), dtype=bool)

        temp_ranges = {'T':(575, 1200), 'Y':(275, 550), 'L':(1300, 2400)}

        for param, val in zip(parameter, value):

            param_use = param_aliases.get(param.lower(), param.lower())

            # common aliases

            if param.lower() in ['teff', 'temp', 'T']:
                if type(value) is str:
                    value = temp_ranges[value.upper()]
                mask = (self.template_parameters['teff'] >= value[0]) & (self.template_parameters['teff'] <= value[1])

    def wavelength_range_of_template(self, library, name):
        path = f'{self.library_path}/{library}/resampled/{name}'
        table = Table.read(path, format='ascii.ecsv', delimiter=' ', names=['wav', 'flux_nu'])
        return table['wav'].min()*u.AA, table['wav'].max()*u.AA


    def add_min_max_wavelengths_to_h5(self, library, output_file_name='photometry_grid.hdf5'):
        mins = []
        maxs = []

        for name in self.template_names[library]:
            min_wav, max_wav = self.wavelength_range_of_template(library, name)
            mins.append(min_wav.to(u.um).value)
            maxs.append(max_wav.to(u.um).value)

            model_file_name = f'{library}_{output_file_name}'
            file_path = f'{self.library_path}/{model_file_name}'

        with h5.File(file_path, 'a') as f:
            f.create_dataset('range/min_wav', data=mins)
            f.create_dataset('range/max_wav', data=maxs)

    def wavelength_range_of_library(self, library):
        min_wav = np.max(self.template_ranges[library], axis=0)[0]
        max_wav = np.min(self.template_ranges[library], axis=0)[1]
        #print(min_wav, max_wav)
        return min_wav, max_wav

    def extreme_wavelength_range_of_library(self, library):
        return np.array(model_wavelength_ranges[library]) * u.um

    def wavelength_range_of_bands(self, bands):
        wav_range = [self.filter_ranges[band] for band in bands]
        wav_range = np.ndarray.flatten(np.array(wav_range)) * u.AA
        return wav_range.min().to(u.um), wav_range.max().to(u.um)


    def create_stellar_interpolator(self, table, method='linear'):
        """
        Create interpolators for stellar parameters from a Table with columns:
        Teff, logg, mass, radius
        
        Args:
            table (astropy.table.Table): Table with columns Teff, logg, mass, radius
            
        Returns:
            tuple: (mass_interpolator, radius_interpolator) - RegularGridInterpolator objects
        """
        # Get unique temperature and log_g values to create the grid
        temp_unique = np.sort(np.unique(table['temp']))
        logg_unique = np.sort(np.unique(table['log_g']))

        # Check if grid is regular
        if len(temp_unique) * len(logg_unique) != len(table):
            print(f"Warning: Data may not form a regular grid! Expected {len(temp_unique) * len(logg_unique)} points, got {len(table)}")

        # Create empty 2D arrays for mass and radius
        mass_grid = np.full((len(temp_unique), len(logg_unique)), np.nan)
        radius_grid = np.full((len(temp_unique), len(logg_unique)), np.nan)

        # Create dictionaries to map values to indices
        temp_to_idx = {temp: i for i, temp in enumerate(temp_unique)}
        logg_to_idx = {logg: i for i, logg in enumerate(logg_unique)}

        # Fill the grids with values
        for row in table:
            i = temp_to_idx[row['temp']]
            j = logg_to_idx[row['log_g']]
            mass_grid[i, j] = row['mass']
            radius_grid[i, j] = row['radius']

        # Delete any rows or columns where less than 10% of the values are filled

        idx_to_remove = []
        for i in range(len(temp_unique)):
            if np.isnan(mass_grid[i]).sum() > 0.9 * len(logg_unique):
                mass_grid[i] = np.nan
                radius_grid[i] = np.nan
                idx_to_remove.append(i)
                print(f"Removing temperature value {temp_unique[i]}")

        temp_unique = np.delete(temp_unique, idx_to_remove)

        idx_to_remove = []
        for j in range(len(logg_unique)):
            if np.isnan(mass_grid[:, j]).sum() > 0.9 * len(temp_unique):
                mass_grid[:, j] = np.nan
                radius_grid[:, j] = np.nan
                idx_to_remove.append(j)
                print(f"Removing log g value {logg_unique[j]}")

        logg_unique = np.delete(logg_unique, idx_to_remove)

        # Delete all all NaN rows and columns
        mass_grid = mass_grid[~np.all(np.isnan(mass_grid), axis=1)]
        mass_grid =  mass_grid[:, ~np.all(np.isnan(mass_grid), axis=0)]

        radius_grid = radius_grid[~np.all(np.isnan(radius_grid), axis=1)]
        radius_grid =  radius_grid[:, ~np.all(np.isnan(radius_grid), axis=0)]

        # Create interpolators for mass and radius
        mass_interpolator = RegularGridInterpolator(
            (temp_unique, logg_unique), mass_grid,
            bounds_error=False, fill_value=None, method = method,
        )

        radius_interpolator = RegularGridInterpolator(
            (temp_unique, logg_unique), radius_grid,
            bounds_error=False, fill_value=None, method = method,
        )

        return mass_interpolator, radius_interpolator, (temp_unique, logg_unique)


    def visualize_grid(self, table, temp_unique, logg_unique, mass_interp, radius_interp, plot_name = 'stellar_parameters_visualization.png'):
        """
        Create visualization of the interpolation grid and results
        """
        # Create a finer grid for visualization
        temp_fine = np.linspace(temp_unique.min(), temp_unique.max(), 300)
        logg_fine = np.linspace(logg_unique.min(), logg_unique.max(), 100)
        temp_grid, logg_grid = np.meshgrid(temp_fine, logg_fine)

        # Interpolate mass and radius on fine grid
        mass_fine, radius_fine = self.interpolate_stellar_parameters(
            temp_grid.flatten(), logg_grid.flatten(), mass_interp, radius_interp
        )
        mass_fine = mass_fine.reshape(temp_grid.shape)
        radius_fine = radius_fine.reshape(temp_grid.shape)

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10), facecolor='white', dpi=200)

        # Plot 1: Mass contour plot
        ax1 = fig.add_subplot(221)
        contour1 = ax1.contourf(temp_fine, logg_fine, mass_fine, 20, cmap='viridis', norm=LogNorm())
        ax1.set_xlabel('Effective Temperature (K)')
        ax1.set_ylabel('log g')
        ax1.set_title('Stellar Mass (M♃) Contour Plot')
        ax1.set_xlim(temp_unique.min(), temp_unique.max())
        ax1.set_ylim(logg_unique.min(), logg_unique.max())
        plt.colorbar(contour1, ax=ax1, label='Mass (M♃)')

        # Add data points to mass plot
        ax1.scatter(table['temp'], table['log_g'], c='red', s=10, alpha=0.5)

        # Plot 2: Radius contour plot
        ax2 = fig.add_subplot(222)
        contour2 = ax2.contourf(temp_fine, logg_fine, radius_fine, 20, cmap='plasma')
        ax2.set_xlabel('Effective Temperature (K)')
        ax2.set_ylabel('log g')
        ax2.set_title('Stellar Radius (R☉) Contour Plot')
        ax2.set_xlim(temp_unique.min(), temp_unique.max())
        ax2.set_ylim(logg_unique.min(), logg_unique.max())
        plt.colorbar(contour2, ax=ax2, label='Radius (R☉)')

        # Plot 3: 3D surface plot for Mass
        ax3 = fig.add_subplot(223, projection='3d')
        surf1 = ax3.plot_surface(temp_grid, logg_grid, mass_fine, cmap='viridis', alpha=0.8)
        ax3.set_xlabel('Effective Temperature (K)')
        ax3.set_ylabel('log g')
        ax3.set_zlabel('Mass (M♃)')
        ax3.set_title('Stellar Mass 3D Surface')

        # Plot 4: 3D surface plot for Radius
        ax4 = fig.add_subplot(224, projection='3d')
        surf2 = ax4.plot_surface(temp_grid, logg_grid, radius_fine, cmap='plasma', alpha=0.8)
        ax4.set_xlabel('Effective Temperature (K)')
        ax4.set_ylabel('log g')
        ax4.set_zlabel('Radius (R☉)')
        ax4.set_title('Stellar Radius 3D Surface')

        plt.tight_layout()
        plt.savefig(plot_name, dpi=300)
        plt.close(fig)

        print(f"Grid visualization saved to {plot_name}")


    def plot_brown_dwarf_locations(self, ra, dec, distances=None,
                            idxs='all', coord_system='galactic', plot_3d=False, ax=None,
                            mw_fit_kwargs={'radius':10 * u.kpc,
                                        'unit':u.kpc,
                                        'coord':"galactocentric",
                                        'annotation':True,
                                        'figsize':(10, 8)},
                             **kwargs):

        assert idxs is not None or distances is not None, 'Provide either an index or a distance.'
        assert type(distances) is u.Quantity if distances is not None else True, 'Distance must be a Quantity.'
        assert coord_system in ['galactic', 'equatorial', 'galactocentric'], 'Coordinate system must be either galactic, galactocentric or equatorial.'
        assert len(ra) == len(dec), 'RA and Dec must be the same length.'

        if idxs == 'all':
            idxs = np.arange(len(ra))

        if distances is None and idxs is not None:
            assert hasattr(self, 'star_min_ix'), 'Fit the catalogue first.'
            assert hasattr(self, 'star_tnorm'), 'Fit the catalogue first.'

            distances = [self.get_physical_params(self.star_min_ix[idx], self.star_tnorm[idx, self.star_min_ix[idx]])['distance'] for idx in idxs]

        assert len(idxs) > 0 and len(ra) > 0 and len(dec) > 0 and len(distances) > 0, 'No valid data provided.'

        fig = None
        if ax is None:
            fig = plt.figure(figsize=(6, 4), dpi=200, facecolor='white')
            if plot_3d:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

        coords_icrs = SkyCoord(ra=ra, dec=dec, distance=distances, frame='icrs')

        if coord_system == 'galactic':
            coords_galactic = coords_icrs.galactic
            ax.scatter(coords_galactic.l, coords_galactic.b, **kwargs)

        elif coord_system == 'galactocentric':
            print(distances)

            coords_galactocentric = coords_icrs.transform_to(Galactocentric)

            dummy = SkyCoord(ra=10*u.deg, dec=10*u.deg, distance=0*u.pc, frame='icrs')
            coords_galactocentric_dummy = dummy.transform_to(Galactocentric)

            if plot_3d:
                ax.scatter(coords_galactocentric.x, coords_galactocentric.y, coords_galactocentric.z, **kwargs)
                ax.scatter(coords_galactocentric_dummy.x, coords_galactocentric_dummy.y, coords_galactocentric_dummy.z, c='yellow', marker='*', label='Sun')

                ax.set_xlabel('X (pc)')
                ax.set_ylabel('Y (pc)')
                ax.set_zlabel('Z (pc)')



                # Add offset circle for thin disk at +- 400 pc
                import mpl_toolkits.mplot3d.art3d as art3d
                from matplotlib.patches import Circle

                disks = [0, 400, -400, 1300, -1300]
                colors = ['b', 'g', 'g', 'r', 'r']
                for thi,c in zip(disks, colors):
                    circ = Circle((0, 0), coords_galactocentric_dummy.galcen_distance.to(u.pc).value, color=c, alpha=1, fill=False)
                    ax.add_patch(circ)
                    art3d.pathpatch_2d_to_3d(circ, z=thi, zdir="z")

            else:

                try:

                    from mw_plot import MWFaceOn
                    plt.close()

                    ax = MWFaceOn(

                        **mw_fit_kwargs
                    )
                    # plot solar system
                      # turn off legend
                    ax.scatter(-1*coords_galactocentric_dummy.x, coords_galactocentric_dummy.y, c='yellow', marker='*', label='Sun')

                    ax.ax.legend().set_visible(False)
                except ImportError:
                    pass

                ax.scatter(-1*coords_galactocentric.x, coords_galactocentric.y, **kwargs)

        return ax



# To Do
# Distances based on normalization - done
# Plotting on galactic coordinates if ra and dec are provided - done
# Filtering templates by stellar class - in progress



class FilterArithmeticParser:
    """
    Parser for filter arithmetic expressions including comparisons.
    Supports operations like:
    - Basic arithmetic: +, -, *, /
    - Comparisons: <, >, <=, >=, ==, !=
    - Parentheses for grouping
    - Constants and coefficients
    
    Examples:
        "F356W + F444W"           -> filter addition
        "F356W > 0.5"             -> boolean mask where filter > 0.5
        "F356W / F444W >= 1.2"    -> boolean mask for color cut
    """

    def __init__(self):
        # Added comparison operators to the map
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }

        # Updated regex:
        # 1. Identifiers: [A-Za-z_][A-Za-z0-9_.]*
        # 2. Numbers: \d+(?:\.\d+)?
        # 3. Multi-char operators: <=, >=, ==, != (Must come before single chars)
        # 4. Single-char operators: <, >, +, -, *, /, (, )
        self.pattern = r'([A-Za-z_][A-Za-z0-9_.]*|\d+(?:\.\d+)?|<=|>=|==|!=|[<>\+\-\*\/\(\)])'

    def tokenize(self, expression: str) -> List[str]:
        """Convert string expression into list of tokens."""
        tokens = re.findall(self.pattern, expression)
        return [token.strip() for token in tokens if token.strip()]

    def is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def is_filter(self, token: str) -> bool:
        """Check if token is a filter name."""
        # Regex matches standard variable names
        return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_.]*$', token))

    def evaluate(self, tokens: List[str], filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Evaluate a list of tokens using provided filter data.
        
        Args:
            tokens: List of tokens from the expression
            filter_data: Dictionary mapping filter names to their values
            
        Returns:
            Result of the arithmetic or comparison operations (float or array)
        """
        output_stack = []
        operator_stack = []

        # Updated precedence:
        # Level 3: Multiplication/Division
        # Level 2: Addition/Subtraction
        # Level 1: Comparisons (lowest priority, evaluated last)
        precedence = {
            '*': 3, '/': 3,
            '+': 2, '-': 2,
            '>': 1, '<': 1, '>=': 1, '<=': 1, '==': 1, '!=': 1
        }

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '(':
                operator_stack.append(token)

            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.pop()  # Remove '('

            elif token in self.operators:
                # While top of stack has greater or equal precedence, apply it
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    self._apply_operator(operator_stack, output_stack, filter_data)
                operator_stack.append(token)

            else:  # Number or filter name
                if self.is_number(token):
                    value = float(token)
                elif self.is_filter(token):
                    if token not in filter_data:
                        raise ValueError(f"Filter {token} not found in provided data")
                    value = filter_data[token]
                else:
                    raise ValueError(f"Invalid token: {token}")
                output_stack.append(value)

            i += 1

        while operator_stack:
            self._apply_operator(operator_stack, output_stack, filter_data)

        if len(output_stack) != 1:
            raise ValueError("Invalid expression")

        return output_stack[0]

    def _apply_operator(self, operator_stack: List[str], output_stack: List[Union[float, np.ndarray]],
                       filter_data: Dict[str, Union[float, np.ndarray]]) -> None:
        """Apply operator to the top two values in the output stack."""
        op_symbol = operator_stack.pop()
        right = output_stack.pop()
        left = output_stack.pop()
        
        # Apply the operator function (e.g., operator.add, operator.lt)
        result = self.operators[op_symbol](left, right)
        output_stack.append(result)

    def parse_and_evaluate(self, expression: str, filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Parse and evaluate a filter arithmetic expression.
        
        Args:
            expression: String containing the filter arithmetic expression
            filter_data: Dictionary mapping filter names to their values
            
        Returns:
            Result of evaluating the expression
        """
        tokens = self.tokenize(expression)
        return self.evaluate(tokens, filter_data)

def iterate_model_parameters(model_parameters: Dict[str, Dict[str, List[Any]]],
                           model_name: str = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Iterate over all parameter combinations for specified model(s) in model_parameters.
    
    Args:
        model_parameters: Nested dictionary of model parameters
        model_name: Specific model to iterate over. If None, iterate over all models.
    
    Returns:
        Iterator yielding tuples of (model_name, parameter_combination)
        where parameter_combination is a dictionary of parameter names and values
    
    Example:
        for model, params in iterate_model_parameters(model_parameters, "sonora_bobcat"):
            print(f"Model: {model}")
            print(f"Temperature: {params['temp']}")
            print(f"Surface gravity: {params['log_gs']}")
            # etc...
    """
    models_to_iterate = [model_name] if model_name else model_parameters.keys()

    for model in models_to_iterate:
        if model not in model_parameters:
            raise ValueError(f"Model {model} not found in model_parameters")

        # Get parameter names and their possible values
        params = model_parameters[model]
        param_names = list(params.keys())
        param_values = list(params.values())

        # Use itertools.product to generate all combinations
        for combination in product(*param_values):
            # Create dictionary of parameter names and their values for this combination
            param_dict = dict(zip(param_names, combination))
            yield model, param_dict

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
