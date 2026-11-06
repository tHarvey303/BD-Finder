"""Empirical ultracool-dwarf library: loading, fitting and the distance/SpT path.

Needs the shipped models/empirical_ucd/ spectra (the photometry grid is rebuilt by
StarFit if missing) and network access for the SVO filter metadata, like the other tests.
"""
import os

import astropy.units as u
import h5py
import numpy as np
import pytest

from BDFit import StarFit

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
BANDS = ['VIS.vis', 'NISP.Y', 'NISP.J', 'NISP.H', 'IRAC.I1', 'IRAC.I2', 'F115W', 'F150W', 'F277W', 'F444W']


@pytest.fixture(scope='module')
def sf():
    return StarFit(libraries=['empirical_ucd'], library_path=LIB_PATH)


def test_library_loads(sf):
    grid = sf.template_grids['empirical_ucd']
    assert grid.shape[0] == len(sf.template_bands['empirical_ucd'])
    assert grid.shape[1] == len(sf.template_names['empirical_ucd']) > 100
    params = sf.template_parameters['empirical_ucd']
    for key in ('temp', 'log_g', 'met', 'co', 'spt', 'spt_str', 'source', 'object_name', 'quality',
                'abs_mag_Y', 'abs_mag_J', 'abs_mag_H', 'coverage_min_um', 'coverage_max_um',
                'extended_blue', 'extended_red'):
        assert key in params, key
    assert sf.template_r_ref['empirical_ucd'] == pytest.approx(0.1)
    with h5py.File(os.path.join(LIB_PATH, 'empirical_ucd_photometry_grid.hdf5'), 'r') as f:
        assert f.attrs['r_ref_rsun'] == pytest.approx(0.1)


def test_fit_synthetic_dwarf(sf):
    lib = 'empirical_ucd'
    params = sf.template_parameters[lib]
    j = int(np.argmin(np.abs(np.asarray(params['spt'], float) - 15.0)))  # an L5 template
    d_true = 250.0 * u.pc
    r_ref = sf.template_r_ref[lib] * u.Rsun
    a = float((r_ref / d_true).decompose()) ** 2
    idx = [sf.model_filters.index(b) for b in BANDS]
    flux = sf.template_grids[lib][idx, j] / sf.scaling_factors[lib] * a  # nJy at d_true
    fnu = np.atleast_2d(flux) * u.nJy
    efnu = 0.05 * fnu
    sf.fit_catalog(fnu=fnu, efnu=efnu, bands=list(BANDS))
    best = int(sf.star_min_ix[0])
    assert best == j
    assert sf.star_min_chi2[0] < 1e-6
    phys = sf.get_physical_params(best, sf.star_tnorm[0, best])
    assert phys['distance'].to_value(u.pc) == pytest.approx(d_true.value, rel=1e-3)
    assert phys['spt'] == pytest.approx(float(params['spt'][j]))
    m_h = float(params['abs_mag_H'][j]) + 5 * np.log10(d_true.value / 10)
    assert phys['app_mag_H'].value == pytest.approx(m_h, abs=1e-3)
    # model libraries keep their behaviour: catalogue-level helper still runs
    distances, masses, radii = sf.get_catalog_physical_params()
    assert distances[0].to_value(u.pc) == pytest.approx(d_true.value, rel=1e-3)

    summary = sf.empirical_ucd_summary(lib, distance_prior='volume')
    assert summary['best_template'][0] == sf.template_names[lib][j]
    assert summary['spt_lo'][0] <= params['spt'][j] <= summary['spt_hi'][0]
    assert np.isfinite(summary['chi2_marg'][0])


def test_irac_tie_is_photometry_only():
    """empirical_ucd carries the DAWN-catalogue IRAC tie on IRAC.I1/I2 only; the spectra (and so
    every other band) are those of empirical_ucd_nisptied."""
    sf2 = StarFit(libraries=['empirical_ucd', 'empirical_ucd_nisptied'], library_path=LIB_PATH)
    g, g0 = sf2.template_grids['empirical_ucd'], sf2.template_grids['empirical_ucd_nisptied']
    bands = sf2.template_bands['empirical_ucd']
    dch1 = np.asarray(sf2.template_parameters['empirical_ucd']['irac_tie_dch1'], float)
    for j, band in enumerate(bands):
        if band == 'IRAC.I1':
            np.testing.assert_allclose(g[j], g0[j] * 10 ** (-0.4 * dch1), rtol=1e-5)
        elif band != 'IRAC.I2':
            np.testing.assert_allclose(g[j], g0[j], rtol=1e-6)
