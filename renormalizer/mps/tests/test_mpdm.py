# -*- coding: utf-8 -*-

import numpy as np
import pytest

from renormalizer.mps import Mps, Mpo, MpDm, MpDmFull, ThermalProp
from renormalizer.model import MolList, Mol, Phonon
from renormalizer.tests import parameter
from renormalizer.utils import Quantity, EvolveConfig, EvolveMethod


@pytest.mark.parametrize("nmols", (2, 3, 4))
@pytest.mark.parametrize("phonon_freq", (0.01, 0.001, 0.0001))
def test_mpdm_full(nmols, phonon_freq):
    ph = Phonon.simple_phonon(Quantity(phonon_freq), Quantity(1), 2)
    m = Mol(Quantity(0), [ph])
    mol_list = MolList([m] * nmols, Quantity(1))

    gs_dm = MpDm.max_entangled_gs(mol_list)
    beta = Quantity(1000, "K").to_beta()
    tp = ThermalProp(gs_dm, Mpo(mol_list), exact=True, space="GS")
    tp.evolve(None, 50, beta / 1j)
    gs_dm = tp.latest_mps
    assert np.allclose(gs_dm.e_occupations, [0] * nmols)
    e_gs_dm = Mpo.onsite(mol_list, r"a^\dagger", mol_idx_set={0}).apply(gs_dm, canonicalise=True)
    assert np.allclose(e_gs_dm.e_occupations, [1] + [0] * (nmols - 1))

    mpdm_full = MpDmFull.from_mpdm(e_gs_dm)
    assert np.allclose(mpdm_full.e_occupations, e_gs_dm.e_occupations)
    assert np.allclose(mpdm_full.ph_occupations, e_gs_dm.ph_occupations, rtol=1e-3)


def test_from_mps():
    gs = Mps.random(parameter.mol_list, 1, 20)
    gs_mpdm = MpDm.from_mps(gs)
    assert np.allclose(gs.e_occupations, gs_mpdm.e_occupations)
    gs = gs.canonicalise()
    gs_mpdm = gs_mpdm.canonicalise()
    assert np.allclose(gs.e_occupations, gs_mpdm.e_occupations)



@pytest.mark.parametrize(
    "mol_list, etot_std, occ_std, rtol",
    (
        # explicit hybrid algorithm will be removed
        #[
        #    parameter.hybrid_mol_list,
        #    0.0853441664951,
        #    [0.20881609, 0.35239430, 0.43878960],
        #    5e-3,
        #],
        [parameter.mol_list, 0.0853413581416, [0.20881782, 0.35239674, 0.43878545], 5e-3],
    ),
)
@pytest.mark.parametrize("adaptive, evolve_method", (
        [True,  EvolveMethod.prop_and_compress],
        [False, EvolveMethod.prop_and_compress],
        [True, EvolveMethod.tdvp_ps],
        [False, EvolveMethod.tdvp_ps],
        [False, EvolveMethod.tdvp_mu_vmf],
))
def test_thermal_prop(mol_list, etot_std, occ_std, adaptive, evolve_method, rtol):
    init_mps = MpDm.max_entangled_ex(mol_list)
    mpo = Mpo(mol_list)
    beta = Quantity(298, "K").to_beta()
    evolve_time = beta / 2j
    
    evolve_config = EvolveConfig(evolve_method, adaptive=adaptive,
                guess_dt=0.1/1j)
    
    if adaptive == True:
        nsteps = 1
    else:
        nsteps = 100
    
    if evolve_method ==  EvolveMethod.tdvp_mu_vmf:
        nsteps = 20
        evolve_config.ivp_rtol=1e-3
        evolve_config.ivp_atol=1e-6
        init_mps.compress_config.bond_dim_max_value=16

    dbeta = evolve_time/nsteps

    tp = ThermalProp(init_mps, mpo, evolve_config=evolve_config)
    tp.evolve(evolve_dt=dbeta, nsteps=nsteps)
    mps = tp.latest_mps
    MPO, HAM, Etot, A_el = mps.construct_hybrid_Ham(mpo, debug=True)
    # exact A_el: 0.20896541050347484, 0.35240029674394463, 0.4386342927525734
    # exact internal energy: 0.0853388060014744
    print("Etot", Etot, A_el)
    assert np.allclose(Etot, etot_std, rtol=rtol)
    assert np.allclose(A_el, occ_std, rtol=rtol)