from astropy.table import Table, Column
import numpy as np
import astropy.units as u
import astropy.constants as c
import glob
from typing import Tuple, List, Dict, Union
import re
import operator
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
import zipfile
import gzip 
import h5py as h5
from numpy.lib.recfunctions import structured_to_unstructured
from astroquery.svo_fps import SvoFps
import matplotlib.patheffects as pe


# Path to download models - currently supports Sonora Bobcat, Cholla, Elf Owl, Diamondback and the Low-Z models
file_urls={'sonora_bobcat':["https://zenodo.org/records/5063476/files/spectra_m+0.0.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m-0.5.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co1.5_g1000nc.tar.gz?download=1", "https://zenodo.org/records/5063476/files/spectra_m+0.0_co0.5_g1000nc.tar.gz?download=1"],
           'sonora_cholla':["https://zenodo.org/records/4450269/files/spectra.tar.gz?download=1"],
           'sonora_evolution_and_photometry':["https://zenodo.org/records/5063476/files/evolution_and_photometery.tar.gz?download=1"],
           'sonora_diamondback':['https://zenodo.org/records/12735103/files/spectra.zip?download=1'],
           'sonora_elf_owl':['https://zenodo.org/records/10385987/files/output_1300.0_1400.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1600.0_1800.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_1900.0_2100.tar.gz?download=1', 'https://zenodo.org/records/10385987/files/output_2200.0_2400.tar.gz?download=1', #Y-type
                            'https://zenodo.org/records/10385821/files/output_1000.0_1200.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_575.0_650.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_850.0_950.tar.gz?download=1', 'https://zenodo.org/records/10385821/files/output_700.0_800.tar.gz?download=1', # T-type
                            'https://zenodo.org/records/10381250/files/output_275.0_325.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_350.0_400.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_425.0_475.0.tar.gz?download=1', 'https://zenodo.org/records/10381250/files/output_500.0_550.0.tar.gz?download=1'], #L-type
           'low-z':['https://dataverse.harvard.edu/api/access/datafile/4571308', 'https://dataverse.harvard.edu/api/access/datafile/4570758']}

# Euclid bands
# "Y", "Blue", "J", "Red", "H", "vis"
# Euclid - NISP
# Paranal - Vista - Z, Y, J, H, Ks

default_bands = ["ACS_WFC.F435W", "ACS_WFC.F475W", "ACS_WFC.F606W", "ACS_WFC.F625W", "ACS_WFC.F775W", "ACS_WFC.F814W", "ACS_WFC.F850LP", 
                "F070W", "F090W", "WFC3_IR.F105W", "WFC3_IR.F110W", "F115W", "WFC3_IR.F125W", "F150W", 
                "WFC3_IR.F140W", "F140M", "WFC3_IR.F160W", "F162M", "F182M", "F200W", "F250M", 
                "F277W", "F300M", "F335M", "F356W", "F360M", "F410M", "F430M",
                "F444W", "F460M", "F480M", "F560W", "F770W", 
                "NISP.Y", "NISP.J", "NISP.H", "VIS.vis",
                "VISTA.Z", "VISTA.Y", "VISTA.J", "VISTA.H", "VISTA.Ks"]

 # "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"]

code_dir = os.path.dirname(os.path.abspath(__file__))

class StarFit:
    def __init__(self, library_path='internal', libraries = ["sonora_bobcat", "sonora_cholla", "sonora_elf_owl", 'sonora_diamondback', 'low-z'], compile_bands='default',
                facilities_to_search={"JWST": ["NIRCam", "MIRI"], "HST": ["ACS", "WFC3"], "Euclid":["NISP", "VIS"], "Paranal":["VIRCAM"]}, resample_step=50, constant_R=False, R=500, min_wav=0.3 * u.micron, 
                max_wav=10 * u.micron, scaling_factor=1e-22):

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
            Scaling factor to shift the models to reasonable flux ranges.
        '''

        if library_path == 'internal':
            self.library_path = os.path.dirname(os.path.dirname(code_dir)) + '/models/'
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

        self.resample_step = resample_step 
        self.constant_R = constant_R
        self.R = R
        self.min_wav = min_wav
        self.max_wav = max_wav
        self.scaling_factor = scaling_factor

        self.template_grids = {}
        self.template_bands = {}
        self.template_names = {}

        if compile_bands == 'default':
            self.model_filters = default_bands
        else:
            self.model_filters = compile_bands
            
        # If compiled data exists, load it. If not, run setup_libraries. 

        for library in libraries:
            if not os.path.exists(f"{self.library_path}/{library}_photometry_grid.hdf5"):
                print(f'Compiled {library} photometry not found.')
                if not os.path.exists(f'{self.library_path}/{library}.param'):
                    print(f'No {library} models found. Running setup_libraries.')
                    self.setup_libraries(self.library_path, [library])
                    self.convert_templates(self.library_path, library)
                
                self.build_template_grid(self.model_filters, model_versions=[library], model_path=self.library_path)
            
            self._load_template_grid(self.library_path, library)

        self.combined_libraries = None

        self._build_combined_template_grid(libraries)


    def _load_template_grid(self, library_path, library):
        with h5.File(f"{library_path}/{library}_photometry_grid.hdf5", 'r') as file:
            self.template_grids[library] = np.array(file['template_grid'][:])
            self.template_bands[library] = list(file.attrs['bands'])
            try:
                self.template_names[library] = list(file.attrs['names'])
            except KeyError:
                if 'names' in file.keys():
                    self.template_names[library] = list(file['names'][:])
                else:
                    file.close()
                    self._fix_names(library, model_path=library_path)
                    
        
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
            assert len(model_filters) == len(idxs[library]), f"Model filters and idxs are different lengths: {len(model_filters)} and {len(idxs[library])}"
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

        if type(wavs_um[0]) == float:
            wavs_um = [i*u.um for i in wavs_um]

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
        model_table = Table.read(f'{model_path}/{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
        assert len(model_table) > 0, f'No models found in {model_version}.param. Maybe remove the file and try again.'

        names = []
        for pos, row in tqdm(enumerate(model_table), total=len(model_table), desc=f'Building {model_version} template grid'):
            name = row["path"].split("/")[-1]
            names.append(name)

        model_file_name = f'{model_version}_{output_file_name}'
            
        with h5.File(model_path+model_file_name, 'a') as file:    
            try:
                file.attrs['names'] = names
            except RuntimeError:
                print('Failed to save names. Likely too long.')
                file.create_dataset('names', data=names)
            file.close()
        

    def build_template_grid(self, bands, model_versions=['sonora_bobcat', 'sonora_cholla'], model_path='/nvme/scratch/work/tharvey/brown_dwarfs/models/', overwrite=False, output_file_name = 'photometry_grid.hdf5'):
    
        exists = False
        for model_version in model_versions:
            model_file_name = f'{model_version}_{output_file_name}'
            
            if os.path.isfile(model_path+model_file_name) and not overwrite:
                exists = True
                file = h5.File(model_path+model_file_name, 'r')
                model_bands = file.attrs['bands']
                file.close()

                # Check all bands are present
                for band in bands:
                    if band not in model_bands:
                        exists = False
            if exists:
                print(f'{model_version} template grid already exists. Skipping.')
            else:
                print(f'Building {model_version} template grid.')
            
            models_table = Table()
            names = []
        
            model_table = Table.read(f'{model_path}/{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
            assert len(model_table) > 0, f'No models found in {model_version}.param. Maybe remove the file and try again.'

            for pos, row in tqdm(enumerate(model_table), total=len(model_table), desc=f'Building {model_version} template grid'):
                name = row["path"].split("/")[-1]

                names.append(name)
                table = Table.read(f'{model_path}/{row["path"]}', names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
                table['flux_njy'] = table['flux_nu'].to(u.nJy)*self.scaling_factor
                convolved_fluxes = [i.value for i in self._convolve_sed(table['flux_njy'], table['wav'].to(u.um), bands, input='flux')]
                assert len(convolved_fluxes) == len(bands), f'Convolved fluxes and bands are different lengths: {len(convolved_fluxes)} and {len(bands)}'
                flux_column = Column(convolved_fluxes, name=name, unit=u.nJy)
                models_table.add_column(flux_column)

            template_grid = models_table.as_array()

            template_grid = structured_to_unstructured(template_grid, dtype=np.float32)

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            with h5.File(model_path+model_file_name, 'w') as file:    
                file.create_dataset('template_grid', data=template_grid, compression='gzip')
                file.attrs['bands'] = bands
                file.attrs['scale'] = self.scaling_factor
                try:
                    file.attrs['names'] = names
                except RuntimeError:
                    print('Failed to save names. Likely too long.')
                    file.create_dataset('names', data=names)

        return template_grid, names

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

    def convert_templates(self, out_folder='sonora_model/', model_version='bobcat', overwrite=False):
        if model_version == 'sonora_bobcat':
            temp = [200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
            log_gs = [10, 17, 31, 56, 100, 178, 316, 562, 1000, 1780, 3160]
            m = ['-0.5', '0.0', '+0.5']
            npermutations = len(temp) * len(log_gs) * len(m)
        elif model_version == 'sonora_cholla':
            temp = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
            log_gs = [31, 56, 100, 178, 316, 562, 1000, 1780, 3162]
            kzzs = [2, 4, 7]
            npermutations = len(temp) * len(log_gs) * len(kzzs)
        elif model_version == 'sonora_elf_owl':
            temp = [275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400]
            log_gs = [17.0, 31.0, 56.0, 100.0, 178.0, 316.0, 562.0, 1000.0, 1780.0, 3160.0]
            mhs =  [-0.5, -1.0, 0.0, 0.5, 0.7, 1.0]
            kzzs = [2.0, 4.0, 7.0, 8.0, 9.0]
            cos = [0.5, 1.0, 1.5, 2.5]
            npermutations = len(temp) * len(log_gs) * len(mhs) * len(kzzs) * len(cos)
        elif model_version == 'sonora_diamondback':
            temp = [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
            log_gs = [31, 100, 316, 1000, 3160]
            ms = ['-0.5', '0.0', '+0.5']
            cos = [1.0]
            fs = ['f1', 'f2', 'f3', 'f4', 'f8', 'nc']
            npermutations = len(temp) * len(log_gs) * len(ms) * len(cos) * len(fs)
        elif model_version == 'low-z':
            log_gs = [3.5, 4.0, 4.5, 5.0, 5.25, 6.0]
            temp = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
            log_Zs = ['+1.0', '+0.75', '+0.25', '+0.5', '+0.0', '-2.5', '-2.0', '-1.5', '-1.0', '-0.25', '-0.5']
            cos = [0.85, 0.1, 0.55]
            kzzs = [-1.0, 10.0, 2.0]
            npermutations = len(log_gs) * len(temp) * len(log_Zs) * len(cos) * len(kzzs)

        count = 1
        if Path(out_folder+f'{model_version}.param').is_file() and not overwrite:
            print('Sonora param file already exists. Skipping.')
            return
        
        try:
            with open(out_folder+f'{model_version}.param', 'w') as f:
                print(f'Total models: ~{npermutations}')
                for temp in tqdm(temp, desc=f'Converting {model_version} models'):
                    for log_g in log_gs:
                        if model_version == 'sonora_bobcat':
                            for metallicity in m:
                                
                                mname = f'_m+{metallicity}/spectra' if metallicity == '0.0' else f'_m{metallicity}'
                                
                                path = f'{self.library_path}/{model_version}/spectra{mname}'
                                name = f'sp_t{temp}g{log_g}nc_m{metallicity}.dat'
                                name_new = f'sp_t{temp}g{log_g}nc_m{metallicity}_resample.dat'
                                if not Path(f'{out_folder}/{model_version}/{name_new}').is_file() or overwrite:
                                    done = self._resample_model(f'{path}/{name[:-4]}', model_version=model_version)
                                    if not done:
                                        print(f'Failed {name}')
                                        continue 
                                    shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                f.writelines(f'{count} {model_version}/{name_new} 1.0\n')
                                count += 1
                                # The special C/O ratio files
                                if log_g == 1000 and metallicity == '0.0':
                                    for co in [0.5, 1.5]:
                                        mname = f"_m+{metallicity}_co{co}_g{log_g}nc"
                                        path = f'{self.library_path}/{model_version}/spectra{mname}'
                                        name = f'sp_t{temp}g{log_g}nc_m+{metallicity}_co{co}.dat'
                                        name_new = f'sp_t{temp}g{log_g}nc_m+{metallicity}_co{co}_resample.dat'
                                        if not Path(f'{out_folder}/{model_version}/{name_new}').is_file() or overwrite:
                                            done = self._resample_model(f'{path}/{name[:-4]}', model_version=model_version)
                                            if not done:
                                                continue 
                                            shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                        f.writelines(f'{count} {out_folder}/{model_version}/{name_new} 1.0\n')
                                        count += 1
                        elif model_version == 'sonora_cholla':
                            for kzz in kzzs:
                                name = f'{temp}K_{log_g}g_logkzz{kzz}.spec'
                                path = f'{self.library}/{model_version}/spectra/spectra_files/'
                                name_new = f'{temp}K_{log_g}g_logkzz{kzz}_resample.dat'
                                if not Path(f'{out_folder}/{model_version}/{name_new}').is_file() or overwrite:
                                    done = self._resample_model(f'{path}/{name[:-5]}', model_version=model_version)
                                    
                                    if not done:
                                        continue 
                                    shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                f.writelines(f'{count} {model_version}/{name_new} 1.0\n')
                                count += 1
                        elif model_version == 'sonora_elf_owl':
                            for mh in mhs:
                                for co in cos:
                                    for kzz in kzzs:
                                        possible_files = os.listdir(f'{self.library_path}/{model_version}/')
                                        possible_files = [i for i in possible_files if os.path.isdir(f'{self.library_path}/{model_version}/{i}')]
                                        options = [(float(i.split('_')[1]), float(i.split('_')[2])) for i in possible_files]
                                        for i in options:
                                            if temp >= i[0] and temp <= i[1]:
                                                temp_low, temp_high = i
                                                break

                                        path = f'{self.library_path}/{model_version}/output_{temp_low}_{temp_high}/'
                                        name = f'spectra_logzz_{kzz}_teff_{float(temp)}_grav_{log_g}_mh_{mh}_co_{float(co)}.nc'
                                        name_new = f'spectra_logzz_{kzz}_teff_{float(temp)}_grav_{log_g}_mh_{mh}_co_{float(co)}_resample.dat'
                                        if not Path(f'{out_folder}/{model_version}/{name_new}').is_file() or overwrite:
                                            done = self._resample_model(f'{path}/{name[:-3]}', model_version=model_version)
                                            if not done:
                                                continue
                                            shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                        f.writelines(f'{count} {model_version}/{name_new} 1.0\n')
                                        count += 1           
                        elif model_version == 'sonora_diamondback':
                            for m in ms:
                                for co in cos:
                                    for fi in fs:
                                        name = f't{temp}g{log_g}{fi}_m{m}_co{co}.spec'
                                        path = f'{self.library_path}/{model_version}/spec/spectra/'
                                        name_new = f't{temp}g{log_g}{fi}_m{m}_co{co}_resample.dat'
                                        if not Path(f'{path}/{name_new}').is_file() or overwrite:
                                            done = self._resample_model(f'{path}/{name[:-5]}', model_version=model_version)
                                            if not done:
                                                continue
                                            shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                        f.writelines(f'{count} {model_version}/{name_new} 1.0\n')
                                        count += 1
                        elif model_version == 'low-z':
                            for log_Z in log_Zs:
                                for co in cos:
                                    for kzz in kzzs:
                                        name = f'LOW_Z_BD_GRID_CLEAR_Teff_{float(temp)}_logg_{float(log_g)}_logZ_{log_Z}_CtoO_{co}_logKzz_{kzz}_spec.txt'
                                        path = f'{self.library_path}/{model_version}/models/models/'
                                        name_new = name[:-4]+'_resample.dat'
                                        if not Path(f'{path}/{name_new}').is_file() or overwrite:
                                            done = self._resample_model(f'{path}/{name[:-4]}', model_version=model_version, resample=False)
                                            if not done:
                                                continue
                                            shutil.copyfile(f'{path}/{name_new}', f'{out_folder}/{model_version}/{name_new}')
                                        f.writelines(f'{count} {model_version}/{name_new} 1.0\n')
                                        count += 1
        except Exception as e:
            if os.path.exists(out_folder+f'{model_version}.param'):
                os.remove(out_folder+f'{model_version}.param')

            raise e
        
        assert count > 1, 'No models found. Check pathing.'
            
    def _resample_model(self, path, model_version, resample=True):
        try:
            if model_version == 'sonora_bobcat':
                with(open(path, 'r')) as f:
                    header = f.readlines()[0]
                    table = Table.read(path, format='ascii', data_start=2, header_start=None, guess=False, fast_reader=False, names=['microns', 'Flux (erg/cm^2/s/Hz)'], units=[u.micron, u.erg/(u.cm**2*u.s*u.Hz)])
            elif model_version == 'sonora_cholla':
                table = Table.read(path+'.spec', format='ascii', data_start=2, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
                header = ''
            elif model_version == 'sonora_elf_owl':
                import xarray
                ds = xarray.load_dataset(path+'.nc')
                wav = ds['wavelength'].data * u.micron
                flux = ds['flux'].data * u.erg/u.cm**2/u.s/u.cm # erg/cm^2/s/cm 
                flux = flux.to(u.erg/u.cm**2/u.s/u.Hz, equivalencies=u.spectral_density(wav))
                wav = wav.to(u.micron)
                table = Table([wav, flux], names=['microns', 'Flux (erg/cm^2/s/Hz)'])
                header = ''
            elif model_version == 'sonora_diamondback':
                table = Table.read(path+'.spec', format='ascii', data_start=3, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
                header = ''
            elif model_version == 'low-z':
                table = Table.read(path+'.txt', format='ascii', data_start=1, header_start=None, guess=False, delimiter='\s', fast_reader=False, names=['microns', 'Watt/m2/m'], units=[u.micron, u.watt/(u.m**2*u.m)])
                table['Flux (erg/cm^2/s/Hz)'] = table['Watt/m2/m'].to(u.erg/(u.cm**2*u.s*u.Hz), equivalencies=u.spectral_density(table['microns'].data*table['microns'].unit))
                header = ''

        except FileNotFoundError as e:
            #print(e)
            return False
        
        table['wav'] = table['microns'].to(u.AA)
        table.sort('wav') 
        table_order = table['wav', 'Flux (erg/cm^2/s/Hz)']
        table_order = table_order[(table['wav'] > self.min_wav) & (table['wav'] < self.max_wav)]
        if len(table_order) == 0:
            return False
            

        spec = Spectrum1D(spectral_axis=u.Quantity(table_order['wav']), flux=u.Quantity(table_order['Flux (erg/cm^2/s/Hz)']))
        new_table = Table()
        
        if resample:
            resample = FluxConservingResampler()
            if self.constant_R:
                new_disp_grid = self._generate_wav_sampling([table_order['wav'].max()]) * u.AA
            else:
                new_disp_grid = np.arange(table_order['wav'].min(), table_order['wav'].max(), self.resample_step) * u.AA
            new_spec = resample(spec, new_disp_grid)
            new_table['wav'] = new_spec.spectral_axis
            new_table['flux_nu'] = new_spec.flux
        else:
            new_table['wav'] = spec.spectral_axis
            new_table['flux_nu'] = spec.flux

        header = [i.replace(',', '') for i in header.split(' ') if i not in ['', '\n', ' ', '(MKS),']]
        new_table.meta = {header[i+len(header)//2]:header[i] for i in range(len(header)//2)}
        new_table.write(path+'_resample.dat', format='ascii.ecsv', overwrite=True)
        return True

    def _generate_wav_sampling(self, max_wavs):
        if type(self.R) not in [list, np.ndarray]:
            R = [self.R]
        else:
            R = self.R
        # Generate the desired wavelength sampling.
        x = [1.]

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
        for i in range(self.NSTAR):
            _m = star_tnorm[:,i:i+1]*star_flux[:,i]
            if subset is None:
                star_chi2[:,i] = (
                    (self.fnu * self.zp - _m)**2 * _wht
                ).sum(axis=1)
            else:
                star_chi2[:,i] = (
                    (self.fnu[subset,:] * self.zp - _m)**2 * _wht
                ).sum(axis=1)
        
        # "Best" stellar template
        star_min_ix = np.argmin(star_chi2, axis=1)
        star_min_chi2 = star_chi2.min(axis=1)
        
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


    def fit_catalog(self, photometry_function, bands, photometry_function_kwargs={}, libraries_to_fit='all', sys_err=None, filter_mask=None, subset=None):
        '''
        Photometry function should be a function that returns the fluxes and flux errors to be fit.

        '''

        fnu, efnu = photometry_function(**photometry_function_kwargs)
        assert len(fnu) == len(efnu), 'Flux and error arrays must be the same length.'
        assert len(fnu[0]) == len(bands), 'Flux and error arrays must have the same number of bands.'
        assert type(fnu) is u.Quantity, 'Fluxes must be astropy Quantities.'
        assert type(efnu) is u.Quantity, 'Errors must be astropy Quantities.'

        self.fnu = fnu.to(u.nJy).value 
        self.efnu = efnu.to(u.nJy).value
        self.NFILT = len(bands)

        # Get template grid here.
        if libraries_to_fit == 'all':
            libraries_to_fit = self.libraries

        self._fetch_transmission_bandpasses()
        
        # Make mask for columns in self.model_filters that aren't in bands

        mask = np.array([band in bands for band in self.model_filters])
        # Check actual bands are in the same order as self.model_filters
        self.bands_to_fit = [band for band in self.model_filters if band in bands]
        idxs = np.array([bands.index(band) for band in self.bands_to_fit])
        self.fnu = self.fnu[:, idxs]
        self.efnu = self.efnu[:, idxs]
        self.zp = np.ones_like(self.fnu)
        self.ok_data = np.ones_like(self.fnu, dtype=bool)
        self.nusefilt = self.ok_data.sum(axis=1)

        # Check that all bands are in the model_filters
        self.reduced_template_grid = self.combined_template_grid[mask, :].T

        result = self._fit_lsq(self.reduced_template_grid, filter_mask=filter_mask, subset=subset, sys_err=sys_err)

    def plot_fit(self, idx, wav_unit=u.micron, flux_unit=u.nJy):

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
        best_ix = self.star_min_ix[idx]
        model_phot = self.star_tnorm[idx, best_ix]*self.reduced_template_grid[best_ix] * u.nJy
        
        plot_model_phot = model_phot.to(flux_unit, equivalencies=u.spectral_density(wavs*wav_unit)).value
        
        ax[0].scatter(wavs, plot_model_phot, label='Best Fit', color='navy')

        ax[0].set_ylabel(f'Flux Density ({flux_unit:latex_inline})')
        ax[1].set_xlabel(f'Wavelength ({wav_unit:latex_inline})')

        ax[1].set_ylabel('Residuals')
        ax[1].scatter(wavs, (flux - model_phot)/flux_err)
        fig.subplots_adjust(hspace=0)

        library, name = self.get_template_name(best_ix)

        ax[0].set_xlim(ax[0].get_xlim())
        ax[0].set_ylim(ax[0].get_ylim())

        self.plot_best_template(best_ix, idx, ax=ax[0], color='navy', wav_unit=wav_unit, flux_unit=flux_unit, linestyle='solid', lw=1)

        info_box = f'Best Fit: {library}:{name}\n$\chi^2_\\nu$: {self.star_min_chinu[idx]:.2f}\n$\chi^2$: {self.star_min_chi2[idx]:.2f}'
        
        ax[0].text(0.05, 0.95, info_box, transform=ax[0].transAxes, fontsize=8, verticalalignment='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])
        ax[1].hlines(0, wavs[0], wavs[-1], linestyle='--', color='k')
        ax[1].vlines(wavs, (flux - model_phot)/flux_err, 0, color='k', alpha=0.5, linestyle='dotted')
        ax[1].set_ylim(np.nanmin((flux - model_phot)/flux_err)-0.2, np.nanmax((flux - model_phot)/flux_err)+0.2)
        ax[0].legend()

        return fig, ax

    def get_template_name(self,idx):
        for library in self.libraries:
            if idx >= self.idx_ranges[library][0] and idx < self.idx_ranges[library][1]:
                return library, self.template_names[library][idx]


    def plot_best_template(self, model_idx, input_idx, ax=None, wav_unit=u.micron, flux_unit=u.nJy, **kwargs):
        library, name = self.get_template_name(model_idx)
        
        path = f'{self.library_path}/{library}/{name}'

        best_fit_coefficients = self.star_tnorm[input_idx, model_idx]

        table = Table.read(path, format='ascii.ecsv', delimiter=' ', names=['wav', 'flux_nu'], units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        table['flux_njy'] = best_fit_coefficients * table['flux_nu'].to(u.nJy) * self.scaling_factor 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
            ax.set_xlabel(f'Wavelength ({wav_unit:latex_inline})')
            ax.set_ylabel(f'Flux Density ({flux_unit:latex_inline})')
        
        ax.plot(table['wav'].to(wav_unit), table['flux_njy'].to(flux_unit), **kwargs)

    def _describe_model(self, path):
        table = Table.read(path, format='ascii.ecsv', delimiter=' ', names=['wav', 'flux_nu'], units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        meta = table.meta
        return meta

    def color_color(x, y, libraries_to_fit='all'):

        grid = self._build_combined_template_grid(libraries_to_fit=libraries_to_fit)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)


        # Create dictionary of views into filter_data with filter names as keys
        filter_idx = {band: i for i, band in enumerate(self.model_filters)}
        filter_data = {band: grid[i] for band, i in filter_idx.items()}

        x_data = parser.parse_and_evaluate(x, filter_data)
        y_data = parser.parse_and_evaluate(y, filter_data)

        ax.scatter(x_data, y_data, c='k', s=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        return fig, ax



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
