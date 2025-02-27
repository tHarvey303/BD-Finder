from astropy.table import Table, Column
import numpy as np
import astropy.units as u
import astropy.constants as c
import glob
from typing import Tuple, List, Dict, Union, Any, Iterator, Optional, Callable
import re
import operator
import collections
import os
from copy import deepcopy
from tqdm import tqdm
import shutil
from pathlib import Path
from matplotlib import pyplot as plt
import spectres 
from scipy.optimize import curve_fit
import requests
from joblib import Parallel, delayed
import tarfile
import zipfile
import gzip 
import h5py as h5
from numpy.lib.recfunctions import structured_to_unstructured
from astroquery.svo_fps import SvoFps
import matplotlib.patheffects as pe
from itertools import product



# Path to download models - currently supports Sonora Bobcat, Cholla, Elf Owl, Diamondback and the Low-Z models
file_urls={'sonora_bobcat':["https://zenodo.org/records/5063476/files/spectra_m+0.0.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m-0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co1.5_g1000nc.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co0.5_g1000nc.tar.gz?download=1"],
           'sonora_cholla':["https://zenodo.org/records/4450269/files/spectra.tar.gz?download=1"],
           'sonora_diamondback':['https://zenodo.org/records/12735103/files/spectra.zip?download=1'],
           'sonora_elf_owl':['https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz?download=1', #Y-type
                            'https://zenodo.org/records/10385821/files/output_1000.0_1200.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_575.0_650.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_850.0_950.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_700.0_800.tar.gz?download=1', # T-type
                            'https://zenodo.org/records/10381250/files/output_275.0_325.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_350.0_400.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_425.0_475.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_500.0_550.0.tar.gz?download=1'], #L-type
           'low-z':['https://dataverse.harvard.edu/api/access/datafile/4571308', 'https://dataverse.harvard.edu/api/access/datafile/4570758']}

evolution_tables = {'sonora_bobcat':["https://zenodo.org/records/5063476/files/evolution_and_photometery.tar.gz?download=1"],
                    'sonora_diamondback':['https://zenodo.org/records/12735103/files/evolution.zip?download=1']}

# in micron
model_wavelength_ranges = {'sonora_elf_owl':(0.6, 15),
                            'sonora_diamondback':(0.3, 250),
                            'sonora_bobcat':(0.4, 50),
                            'sonora_cholla':(0.3, 250),
                            'low-z':(0.1, 99)}
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
}

model_param_names = ["temp", "log_g", "kzz", "met", "co", "f"]
model_param_dtypes = {
    "temp": float,
    "log_g": float,
    "kzz": float,
    "met": str,
    "co": str,
    "f": str
}

# elf owl
# The parameters included within this grid are effective temperature (Teff), gravity (log(g)), vertical eddy diffusion coefficient (log(Kzz)), atmospheric metallicity ([M/H]), and Carbon-to-Oxygen ratio (C/O).


default_bands = ["ACS_WFC.F435W", "ACS_WFC.F475W", "ACS_WFC.F606W", "ACS_WFC.F625W", "ACS_WFC.F775W", "ACS_WFC.F814W", "ACS_WFC.F850LP", 
                "F070W", "F090W", "WFC3_IR.F105W", "WFC3_IR.F110W", "F115W", "WFC3_IR.F125W", "F150W", 
                "WFC3_IR.F140W", "F140M", "WFC3_IR.F160W", "F162M", "F182M", "F200W", "F210M", "F250M", 
                "F277W", "F300M", "F335M", "F356W", "F360M", "F410M", "F430M",
                "F444W", "F460M", "F480M", "F560W", "F770W", 
                "NISP.Y", "NISP.J", "NISP.H", "VIS.vis",
                "VISTA.Z", "VISTA.Y", "VISTA.J", "VISTA.H", "VISTA.Ks"]

 # "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"]

code_dir = os.path.dirname(os.path.abspath(__file__))

class StarFit:
    def __init__(self, library_path='internal', libraries = ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback', 'low-z'], compile_bands='default',
                facilities_to_search={"JWST": ["NIRCam", "MIRI"], "HST": ["ACS", "WFC3"], "Euclid":["NISP", "VIS"], "Paranal":["VIRCAM"]}, resample_step=50, constant_R=True, R=300, min_wav=0.3 * u.micron, 
                max_wav=12 * u.micron, scaling_factor=1e-22, verbose=False):

        '''

        Parameters
        ----------
        library_path : str
            Path to the model libraries. Default is 'models/'.
        libraries : list
            List of model libraries to compile. Default is ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback'].
        compile_bands : list
            List of bands to compile the models for. Default is 'default', which uses the default_bands.
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
        scaling_factor : float
            Scaling factor to shift the models to reasonable flux ranges. In general use this should not be modified, as changing this
            without also recomputing all photometry will result in incorrectly normalized SEDs. Default is 1e-22.
        '''

        if library_path == 'internal':
            self.library_path = os.path.dirname(os.path.dirname(code_dir)) + '/models'
        else:
            self.library_path = library_path   

        print(f'Library path: {self.library_path}')

        self.libraries = libraries

        self.band_codes = {}
        self.filter_wavs = {}
        self.filter_ranges = {}
        self.filter_instruments = {}
        self.transmission_profiles = {}
        self.facilities_to_search = facilities_to_search
        self.model_parameters = model_parameters
        self.verbose = verbose

        self.resample_step = resample_step 
        self.constant_R = constant_R
        self.R = R
        self.min_wav = min_wav
        self.max_wav = max_wav
        self.scaling_factor = scaling_factor

        self.template_grids = {}
        self.template_bands = {}
        self.template_names = {}
        self.template_parameters = {}

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

            if 'meta' in file:
                if library not in self.template_parameters.keys():
                    self.template_parameters[library] = {}
                for key in list(file['meta'].keys()):
                    self.template_parameters[library][key] = file['meta'][key][:]
                    # parse string metas if needed
                    if any([isinstance(i, bytes) for i in self.template_parameters[library][key]]):
                        self.template_parameters[library][key] = [i.decode('utf-8') for i in self.template_parameters[library][key]]

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
            assert len(self.model_filters) == len(idxs[library]), f"Model filters and idxs are different lengths: {len(model_filters)} and {len(idxs[library])}"
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

    def setup_libraries(self, path='sonora_data/', libraries = ["sonora_bobcat", "sonora_cholla", "sonora_evolution_and_photometry"]):
        for library in libraries:
            # Ensure the destination folder exists, or create it if not
            new_path = path + f"/{library}/"
    
            os.makedirs(new_path, exist_ok=True)

            # Fetch the Zenodo metadata for the given repository URL
            for file_url in file_urls[library]:
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
                    file_name = response.headers['Content-Disposition'].split("attachment; filename*=UTF-8''")[-1]

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
        
    def build_template_grid(self, bands, model_versions=['sonora_bobcat', 'sonora_cholla'], model_path='/nvme/scratch/work/tharvey/brown_dwarfs/models/', overwrite=False, output_file_name='photometry_grid.hdf5'):
        
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
                    table['flux_njy'] = table['flux_nu'].to(u.nJy)*self.scaling_factor
                    
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
                    table['flux_njy'] = table['flux_nu'].to(u.nJy)*self.scaling_factor

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
                    file.attrs['scale'] = self.scaling_factor
                    
                    file.create_dataset('names', data=names)
                    # Stores meta for dataset - temperature, log_g, met, etc. Useful for filtering models, calculating distances, etc.
                    file.create_group('meta')
                    for key in metas.keys():
                        file['meta'][key] = metas[key]

        self.template_grids[model_version] = template_grid
        self.template_bands[model_version] = bands
        self.template_names[model_version] = names
        self.template_parameters[model_version] = metas.keys()

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

            assert len(filter_wavs) == len(self.model_filters), f"Missing filters: {set(self.model_filters) - set(filter_wavs.keys())}"

            self.band_codes = band_codes
            self.filter_wavs = filter_wavs
            self.filter_ranges = filter_ranges
            self.filter_instruments = filter_instruments
            self.pivot = np.array([self.filter_wavs[band].to(u.AA).value for band in self.model_filters]) 
            
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
        model_ext = {'sonora_bobcat':'', 'sonora_cholla':'.spec', 'sonora_elf_owl':'.nc', 'sonora_diamondback':'.spec', 'low-z':'.txt'}
        
        return model_ext[model_version]

    def _latex_label(self, param):
        self.latex_labels = { 'temp': r'$T_{\rm eff}$', 'log_g': r'$\log g$', 'met': r'$\rm [Fe/H]$', 'kzz': r'$\rm K_{zz}$', 'co': r'$\rm C/O$', 'f':r'$\rm fsed'}
        return self.latex_labels[param]

    def param_unit(self, param):
        self.units = {'temp': u.K, 'log_g': u.m/u.s**2, 'met': u.dimensionless_unscaled, 'kzz': u.dimensionless_unscaled, 'co': u.dimensionless_unscaled, 'f':u.dimensionless_unscaled}
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
                                       
                else:
                    raise Exception(f'Unknown model version: {model_version}')
                
                if f'{model_version}/resampled/{name_new}' not in processed_files:
                    f.writelines(f'{model_version}/resampled/{name_new}\n')
                all_file_names.remove(name)
                count += 1
        
        if len(all_file_names) > 0:
            for file in all_file_names:
                if self.verbose:
                    print('Failed to convert:', file)

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
                table = Table.read(path, format='ascii', data_start=2, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
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
                table = Table.read(path, format='ascii', data_start=3, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
            elif model_version == 'low-z':
                table = Table.read(path, format='ascii', data_start=1, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
        except FileNotFoundError as e:
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

        # Least squares normalization of stellar templates
        if subset is None:
            _wht = 1/(self.efnu**2+(sys_err*self.fnu)**2)
        
            _wht /= self.zp**2
            _wht[(~self.ok_data) | (self.efnu <= 0)] = 0

        else:
            _wht = 1.0 / (
                self.efnu[subset,:]**2 + (sys_err * self.fnu[subset,:])**2
            )

            _wht /= self.zp**2
            _wht[(~self.ok_data[subset,:]) | (self.efnu[subset,:] <= 0)] = 0

        # if we want to use this, will need to provide grid just for subset templates bands
        '''clip_filter = (self.pivot < wave_lim[0]) | (self.pivot > wave_lim[1])
        if filter_mask is not None:
                clip_filter &= filter_mask
                
        _wht[:, clip_filter] = 0'''
            
        if subset is None:
            _num = np.dot(self.fnu * self.zp * _wht, star_flux)
        else:
            _num = np.dot(self.fnu[subset,:] * self.zp * _wht, star_flux)
            
        _den= np.dot(1*_wht, star_flux**2)
        _den[_den == 0] = 0
        star_tnorm = _num/_den
        
        # Chi-squared
        star_chi2 = np.zeros(star_tnorm.shape, dtype=np.float32)
        for i in tqdm(range(self.NSTAR), desc='Calculating normalization and chi2 for all templates...'):
            _m = star_tnorm[:,i:i+1]*star_flux[:,i]
            if subset is None:
                star_chi2[:,i] = (
                    (self.fnu * self.zp - _m)**2 * _wht
                ).sum(axis=1)
            else:
                star_chi2[:,i] = (
                    (self.fnu[subset,:] * self.zp - _m)**2 * _wht
                ).sum(axis=1)
        
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
        assert len(fnu[0]) == len(bands), 'Flux and error arrays must have the same number of bands.'
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
                    fitted_bands.append(full_band)
                    band_idx.append(match_idx)
                else:
                    
                    print(f'Warning! Band {band} not in model_filters. Removing from bands to fit.')
                    bands.remove(band)
            else:
                fitted_bands.append(band)
                band_compar_dict[band] = band
                band_idx.append(self.model_filters.index(band))

        min_wav, max_wav = self.wavelength_range_of_bands(fitted_bands)
        check_band_ranges = False

        extreme_min_wav, extreme_max_wav = 0 * u.um, np.inf * u.um
        for library in libraries_to_fit:
            library_min_wav, library_max_wav = self.wavelength_range_of_library(library)
            library_extreme_min_wav, library_extreme_max_wav = self.extreme_wavelength_range_of_library(library)

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

        
        grid = self._build_combined_template_grid(libraries_to_fit)

        print(f'Fitting with {", ".join(libraries_to_fit)} libraries with {len(grid)} templates.')

        mask = np.array([i in fitted_bands for i in self.model_filters])
        # Check actual bands are in the same order as self.model_filters
        self.bands_to_fit = fitted_bands
        # Make sure fnu and efnu are in the same order as self.model_filters - don't need same position, just same order
        idxs = [self.model_filters.index(band) for band in fitted_bands]
        # convert to order - e.g. 0, 1, 2, 3, 4, 5 -> 0, 1, 2, 3, 4, 5
        idxs = np.argsort(idxs)

        print(f'Fitting {len(fitted_bands)} bands: {fitted_bands}')

        self.NFILT = len(fitted_bands)
        self.fnu = self.fnu[:, idxs]
        self.efnu = self.efnu[:, idxs]
        self.zp = np.ones_like(self.fnu)
        self.ok_data = np.ones_like(self.fnu, dtype=bool)
        self.nusefilt = self.ok_data.sum(axis=1)
        self.mask = mask

        assert len(self.bands_to_fit) == self.NFILT, f'Number of bands to fit does not match number of bands in fluxes: {len(self.bands_to_fit)} != {self.NFILT}'

        # Check that all bands are in the model_filters
        self.reduced_template_grid = grid[self.mask, :].T

        fit_results = self._fit_lsq(self.reduced_template_grid, filter_mask=filter_mask, subset=subset, sys_err=sys_err)

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
                pickle.dump(dump_dict, f)

        return fit_results

    def load_results_from_pickle(self, path):
        import pickle
        with open(path, 'rb') as f:
            results = pickle.load(f)

        for key in results.keys():
            if key == 'mask':
                self.reduce_template_grid = self.combined_template_grid[results[key], :].T

            setattr(self, key, results[key])
 
    def plot_fit(self, idx=None, cat_id=None, wav_unit=u.micron, flux_unit=u.nJy):

        assert idx is not None or cat_id is not None, 'Provide either an index or a catalogue id. (if catalogue IDs were provided during fitting'

        if cat_id is not None and idx is not None:
            raise Exception('Provide either an index or a catalogue id, not both.')
        
        if cat_id is not None:
            pname = cat_id
            idx = np.where(self.catalogue_ids == cat_id)[0][0]
        else:
            pname = idx
            

        wavs = [self.filter_wavs[band].to(wav_unit).value for band in self.bands_to_fit]
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

        ax[0].errorbar(wavs, plot_flux, yerr=err, fmt='o', label='Data',
                        color='crimson', markeredgecolor='k', zorder=10)

        if flux_unit == u.ABmag:
            
            # plot any negative fluxes as  3 sigma upper limits
            for i, f in enumerate(flux):
                if f < 0:
                    from matplotlib.patches import FancyArrowPatch
                    error_3sigma = 3*flux_err[i]
                    error_3sigma = error_3sigma.to(flux_unit, equivalencies=u.spectral_density(wavs[i]*wav_unit)).value
                    print(error_3sigma)
                    length = abs(ax[0].get_ylim()[1] - ax[0].get_ylim()[0])*0.15
                    ax[0].add_patch(FancyArrowPatch((wavs[i], error_3sigma), (wavs[i], error_3sigma+length), color='crimson', zorder=10, edgecolor='k', arrowstyle='-|>, scaleA=3', mutation_scale=2))

        best_ix = self.star_min_ix[idx]

        if best_ix != -1:
            print(f"No best fit found for {idx=}!")
            model_phot = self.star_tnorm[idx, best_ix] * self.reduced_template_grid[best_ix] * u.nJy
            plot_model_phot = model_phot.to(flux_unit, equivalencies=u.spectral_density(wavs*wav_unit)).value
            ax[0].scatter(wavs, plot_model_phot, label='Best Fit', color='navy')
            ax[1].scatter(wavs, (flux - model_phot) / flux_err)
            library, name = self.get_template_name(best_ix)
            self.plot_best_template(best_ix, idx, ax=ax[0], color='navy', wav_unit=wav_unit, flux_unit=flux_unit, linestyle='solid', lw=1)
            params = self._extract_model_meta(library, name)
            latex_labels = [self._latex_label(param) for param in params.keys()]
            info_box = '\n'.join([f'{latex_labels[i]}: {params[param]}{self.param_unit(param):latex}' for i, param in enumerate(params.keys())])
            lower_info_box = f'$\chi^2_\\nu$: {self.star_min_chinu[idx]:.2f}\n$\chi^2$: {self.star_min_chi2[idx]:.2f}'
            info_box = f'{pname}\nBest Fit: {library.replace("_", " ").capitalize()}\n{info_box}\n{lower_info_box}'
            ax[0].text(1.02, 0.98, info_box, transform=ax[0].transAxes, fontsize=8, verticalalignment='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')], bbox=dict(facecolor='w', alpha=0.5, edgecolor='black', boxstyle='square,pad=0.5'))
            ax[1].vlines(wavs, (flux - model_phot) / flux_err, 0, color='k', alpha=0.5, linestyle='dotted')
            ax[1].set_ylim(np.nanmin((flux - model_phot) / flux_err) - 0.2, np.nanmax((flux - model_phot)/flux_err)+0.2)
        
        ax[0].set_ylabel(f'Flux Density ({flux_unit:latex_inline})')
        ax[1].set_xlabel(f'Wavelength ({wav_unit:latex_inline})')
        ax[1].set_ylabel('Residuals')
        fig.subplots_adjust(hspace=0)
        ax[0].set_xlim(ax[0].get_xlim())
        ax[0].set_ylim(ax[0].get_ylim())
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
            "scaling_factor": self.scaling_factor,
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
    ):
        # save as .h5
        if not save_path[-3:] == ".h5":
            save_path = f"{'.'.join(save_path.split('.')[:-1])}.h5"
        
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
                libraries_IDs[library].extend([idx])
                libraries_SEDs[library].extend(
                    [self.load_SED(self.star_min_ix[idx], idx, wav_unit = wav_unit, flux_unit = flux_unit)]
                )

        with h5.File(save_path, "w") as f:

            for library in unique_libraries:
                consistent_wavelengths = all(
                    all(
                        np.array(libraries_SEDs[library][0][0].to(wav_unit).value)
                        == np.array(library_SEDs[0].to(wav_unit).value)
                    )
                    for library_SEDs in libraries_SEDs[library]
                )
                consistent_flux_lengths = all(
                    len(libraries_SEDs[library][0][1]) == len(library_SEDs[1])
                    for library_SEDs in libraries_SEDs[library]
                )
                # ensure all wavelengths are the same
                assert consistent_wavelengths
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
        table['flux_njy'] = best_fit_coefficients * table['flux_nu'].to(u.nJy) * self.scaling_factor
        
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

    def color_color(self, x, y, libraries='all', unit=u.ABmag, show_fitted_galaxies=False, color_by=None, **kwargs):

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
        filter_data = {band: grid_converted[i] for band, i in filter_idx.items()}
        
        parser = FilterArithmeticParser()

        x_data = parser.parse_and_evaluate(x, filter_data)
        y_data = parser.parse_and_evaluate(y, filter_data)

        if color_by is not None:
            pass

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
        
        ax.text(0.05, 0.95, f'Libraries:\n{libraries_string}', transform=ax.transAxes, fontsize=8, verticalalignment='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

        return fig, ax
    
    def filter_template_grid_by_parameter(self, parameter, value, exclude_unparametrized=True, combine='and'):
        '''
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
    
    def wavelength_range_of_library(self, library):
        name = self.template_names[library][0]
        min_wav, max_wav = self.wavelength_range_of_template(library, name) 
        return min_wav, max_wav

    def extreme_wavelength_range_of_library(self, library):
        return np.array(model_wavelength_ranges[library]) * u.um

    def wavelength_range_of_bands(self, bands):
        wav_range = [self.filter_ranges[band] for band in bands]
        wav_range = np.ndarray.flatten(np.array(wav_range)) * u.AA
        return wav_range.min().to(u.um), wav_range.max().to(u.um)

        
        
# To Do
# Distances based on normalization
# Plotting on galactic coordinates if ra and dec are provided
# Filtering templates by stellar class, 



class FilterArithmeticParser:
    """
    Parser for filter arithmetic expressions.
    Supports operations like:
    - Basic arithmetic: +, -, *, /
    - Parentheses for grouping
    - Constants and coefficients
    
    Examples:
        "F356W"                    -> single filter
        "F356W + F444W"           -> filter addition
        "2 * F356W"               -> coefficient multiplication
        "(F356W + F444W) / 2"     -> average of filters
        "F356W - 0.5 * F444W"     -> weighted subtraction
    """
    
    def __init__(self):
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        
        # Regular expression pattern for tokenizing
        self.pattern = r'(\d*\.\d+|\d+|[A-Za-z]\d+[A-Za-z]+|\+|\-|\*|\/|\(|\))'
    
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
        return bool(re.match(r'^[A-Za-z]\d+[A-Za-z]+$', token))
    
    def evaluate(self, tokens: List[str], filter_data: Dict[str, Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Evaluate a list of tokens using provided filter data.
        
        Args:
            tokens: List of tokens from the expression
            filter_data: Dictionary mapping filter names to their values
            
        Returns:
            Result of the arithmetic operations
        """
        output_stack = []
        operator_stack = []
        
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        
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
        operator = operator_stack.pop()
        b = output_stack.pop()
        a = output_stack.pop()
        output_stack.append(self.operators[operator](a, b))
    
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