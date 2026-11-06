[![DOI](https://zenodo.org/badge/712418272.svg)](https://doi.org/10.5281/zenodo.18404281)

# BD Fitter


This python package is for fitting brown dwarf (substellar mass object) photometry.  It currently supports the Sonora Bobcat, Cholla, Diamondback and Elf Owl models, and the LOW-Z and ATMO2020 models.
Photometry for a range of filters is provided, and if more are needed the models are automatically downloaded from Zenodo/Harvard DataVerse if not found in directory code is run in.

Model links: 

[Sonora Bobcat](https://zenodo.org/records/5063476) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac141d))

[Sonora Cholla](https://zenodo.org/records/4450269) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac3140/meta))

[Sonora Diamondback](https://zenodo.org/records/12735103) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad71d5/meta))

Sonora Elf Owl ([Y-Type](https://zenodo.org/records/10381250), [T-Type](https://zenodo.org/records/10385821), [L-Type](https://zenodo.org/records/10385987), [Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad18c2/meta))

[LOWZ](https://doi.org/10.7910/DVN/SJRXUO) ([Paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac013c))

[ATMO2020](https://perso.ens-lyon.fr/isabelle.baraffe/ATMO2020/) ([Paper](https://www.aanda.org/articles/aa/full_html/2020/05/aa37381-19/aa37381-19.html))

An optional PHOENIX synthetic stellar (not sub-stellar) library is also supported, for use as a star/contaminant template set rather than for brown dwarf fitting itself. It is not included by default (`StarFit(libraries=['PHOENIX'])` to opt in) since it is a ~9GB download.

[PHOENIX](https://zenodo.org/records/17674612) ([Paper](https://doi.org/10.1051/0004-6361/201219058))

Please cite all fitted model papers and repositories DOIs if you use this code, this would not be possible without the hard work of the authors of these models.

## Installation

Clone this repository into a folder, and then do pip install . to install. 

An example of use is available in examples/chi2_testing.ipynb. All the user needs to provide is an array of flux and flux uncertainity, and a list of bands (in SVO filter format, or abbreviated will work if HST/NIRcam filters).

Code is still a work in progress. Contact the author at: thomas.harvey-3@manchester.ac.uk for help, improvements, problems, bugs, complaints etc...


## Empirical ultracool-dwarf libraries

Besides the model atmospheres, BDFit ships empirical ultracool-dwarf (UCD) template libraries. Each
template is one observed dwarf, from M6 to Y1. They were built for the Euclid Deep Field brown-dwarf work
because model grids get real dwarf colours wrong: Bobcat is ~0.9 mag too bright in Euclid VIS, model
J-H is too blue, and model [3.6]-[4.5] is 0.5-1.2 mag too red at the L/T transition.

There are three versions, each with a field and a subdwarf library. Field = 134 templates; subdwarfs =
28 d/sd, sd and esd templates.

| library | spectra | IRAC.I1 / IRAC.I2 band fluxes | use |
|---|---|---|---|
| `empirical_ucd`, `empirical_ucd_sd` (default) | tied to the Euclid DR1 dwarf locus | tied to the DAWN catalogue system | fitting DAWN / Euclid Deep Field catalogues |
| `empirical_ucd_nisptied`, `empirical_ucd_sd_nisptied` | tied to the Euclid DR1 dwarf locus | synthetic | Euclid photometry with standard Spitzer IRAC |
| `empirical_ucd_synth`, `empirical_ucd_sd_synth` | untied | synthetic | pure synthetic photometry of the spectra |

**Sources** (data only; full provenance per template in `models/<library>/<library>_provenance.ecsv`):
- SpeX Prism Library spectra, 0.65-2.5 um, from the SPLAT data directory. These are the
  Kirkpatrick+10 / Burgasser+06 dwarf standards, the d/sd, sd and esd standards, and up to four more
  field dwarfs per subtype.
- JWST/NIRSpec PRISM spectral standards of Pritchard et al. 2026 (RNAAS 10, 144), using their DAWN JWST
  Archive v4 extractions. The journal supplement holds 17 identical files.
- Beiler et al. 2024 (ApJ 973, 107) NIRSpec + MIRI spectra of late-T and Y dwarfs (Zenodo 14976898).
- Absolute scale: the object's own parallax where available, otherwise the Sanghi et al. 2024 M_JE(SpT)
  relation.
- Colour anchors and ties:
  - Žerjal et al. 2025 (Euclid Q1) and Leggett, Phillips & Tremblin 2025: I_E-Y_E.
  - Kirkpatrick et al. 2021: IRAC colours of model-extended spectra.
  - The Euclid DR1 locus of spectroscopically typed Q1 dwarfs: Y_E-J_E, J_E-H_E, I_E-H_E, and the IRAC
    tie.
  - COSMOS-Web NIRCam-confirmed late T dwarfs: the late-T IRAC tie.
- Models only fill gaps outside each spectrum's coverage. Blueward: ATMO2020, Sonora Bobcat or
  PHOENIX. Redward: the nearest-type JWST spectrum where one exists, otherwise Sonora Elf Owl or PHOENIX.

**Flux convention.** A template stores F_nu(10 pc) / (R_ref / 10 pc)^2, with R_ref = 0.1 Rsun (grid
attribute `r_ref_rsun`). The observed flux at distance d is `template_grid / scale * (R_ref / d)^2`. A
fitted normalisation `norm` therefore gives the distance through the template's own absolute
magnitude: `d = R_ref / sqrt(norm * scale)`. R_ref is a bookkeeping constant, not a physical radius.

**Meta columns** (`StarFit.template_parameters[library]` and `meta/` in the grid):

| column | meaning |
|---|---|
| `temp` | Teff from SpT: Filippazzo+15 for M6-T9, Kirkpatrick+21 for Y |
| `log_g` | 5.0 (dummy) |
| `met` | '0.0' for field dwarfs, 'sd' for subdwarfs |
| `co` | '' |
| `spt` | numeric SpT: M6 = 6, L0 = 10, T0 = 20, Y0 = 30 |
| `spt_str` | type string; subdwarfs prefixed d/sd, sd or esd |
| `subclass` | field, d/sd, sd or esd |
| `source`, `object_name` | origin of the spectrum |
| `quality`, `quality_snr` | A/B/C grade from the median S/N over 1.0-1.7 um, and that S/N |
| `abs_mag_Y`, `abs_mag_J`, `abs_mag_H`, `abs_mag_IE` | Euclid AB absolute magnitudes at 10 pc |
| `coverage_min_um`, `coverage_max_um` | empirical wavelength coverage |
| `extended_blue`, `extended_red` | whether the spectrum was extended outside that coverage |
| `ext_blue_source`, `ext_red_source` | what was used for each extension |
| `scale_source` | how the absolute scale was set |
| `vis_corr_mag` | VIS correction applied |
| `irac_recal_dch1`, `irac_recal_dch2` | IRAC recalibration of model-extended spectra |
| `locus_tied`, `locus_dYJ`, `locus_dJH`, `locus_dIH` | DR1 locus tie of the spectra |
| `irac_tied`, `irac_tie_dch1`, `irac_tie_dch2` | DAWN IRAC tie: mag added to the IRAC.I1 / IRAC.I2 band flux |
| `r_ref_rsun` | R_ref in Rsun |

**Loading and fitting.**

```python
from BDFit import StarFit
sf = StarFit(libraries=['empirical_ucd'])          # add 'empirical_ucd_sd' for subdwarfs
sf.fit_catalog(fnu=fnu, efnu=efnu, bands=bands)    # astropy Quantities, as for the model libraries
ix = sf.star_min_ix[0]
phys = sf.get_physical_params(ix, sf.star_tnorm[0, ix])
```

- For these libraries `get_physical_params` returns `distance`, `spt`, `spt_str`, `met`, `template`,
  `abs_mag_Y/J/H` and the implied `app_mag_H`. `mass` and `radius` are `None`.
- `sf.empirical_ucd_summary('empirical_ucd', distance_prior='none'|'volume', d_max_pc=None)` is optional.
  It returns, per object: the best template, best SpT and distance; the SpT range within Δχ²; the
  weighted mean and std of SpT; the median distance; and a template-marginalised χ². Existing outputs
  are unchanged.
- The model libraries behave exactly as before.

**Rebuilding the grids.** The spectra ship in `models/<library>/spectra/*.ecsv`: wav in Angstrom,
flux_nu in erg/s/cm2/Hz, with the meta in the ECSV header. They are marked `'local'` in `file_urls`, so
nothing is downloaded. To rebuild, delete `models/empirical_ucd*.param`,
`models/empirical_ucd*_photometry_grid.hdf5` and `models/empirical_ucd*/resampled`, then instantiate
`StarFit(libraries=[...])`. This resamples the spectra and runs `build_template_grid`, which also applies
the photometry-only IRAC tie from the template meta (`EMPIRICAL_BAND_TIES`). The spectra themselves are
produced by `bd_v2/build_empirical_templates.py`, which lives in the Euclid analysis scripts, not in
this repository.

**Caveats.**
- Euclid I_E (0.55-0.92 um) always contains a model piece. SpeX starts at 0.65 um and JWST PRISM at
  0.65-1.0 um, so I_E is calibrated to the observed colours above. There is no I_E anchor for Y1 or for
  subdwarfs.
- The IRAC tie in `empirical_ucd[_sd]` is a DAWN-catalogue photometric-system correction: point-source
  IRAC in that catalogue is 0.4-0.9 mag faint relative to H. It is applied to the IRAC.I1 / IRAC.I2 band
  fluxes only, not to the spectra, so NIRCam F277W/F356W/F444W and all other bands keep their physical
  values. A plotted best-fit spectrum therefore sits off its own IRAC grid points. Use `*_nisptied` for
  photometry on a standard Spitzer system.
- For L4-T1 the 3-5 um shape is from Sonora Elf Owl, with only the IRAC band integrals calibrated per
  type. M6-M8, L0-L3 and T2 onwards use JWST spectra there.
- Subdwarfs are sparse beyond L7. 21 of them use the field M_JE relation. Bands extending beyond 5.2 um
  (F560W, F770W) hold partial values; StarFit masks them by template range.
- The `.param` reader in `build_template_grid` (`Table.read(..., format='ascii', names=['path'])`)
  consumes the first line as a header row, so every model library has one template fewer than its
  `.param` file lists. The empirical `.param` files start with a `path` header line to avoid this. The
  model libraries are not changed.

**Test.** `pytest tests/test_empirical_ucd.py` checks three things:
- the library loads with its meta and `r_ref_rsun`;
- a noise-free synthetic L5 dwarf at 250 pc is fitted back to its own template, SpT and distance,
  including `empirical_ucd_summary()`;
- the IRAC tie changes only IRAC.I1 and IRAC.I2.
