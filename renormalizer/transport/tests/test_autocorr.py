# -*- coding: utf-8 -*-

import numpy as np
import qutip
import pytest

from renormalizer.model import Phonon, Mol, MolList, MolList2
from renormalizer.transport.autocorr import TransportAutoCorr
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig, EvolveMethod, CompressCriteria
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_hamiltonian, get_qnidx


@pytest.mark.parametrize("scheme", (
        3,
        4,
))
@pytest.mark.parametrize("mollist2", (
        True,
        False,
))
def test_autocorr(scheme, mollist2):
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph])
    mol_list1 = MolList([mol] * 5, Quantity(1), scheme)
    if mollist2:
        mol_list = MolList2.MolList_to_MolList2(mol_list1)
    else:
        mol_list = mol_list1
    temperature = Quantity(50000, 'K')
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True, guess_dt=0.5, adaptive_rtol=1e-3)
    ievolve_config = EvolveConfig(EvolveMethod.tdvp_ps, adaptive=True, guess_dt=-0.1j)
    ac = TransportAutoCorr(mol_list, temperature, compress_config=compress_config, ievolve_config=ievolve_config, evolve_config=evolve_config)
    ac.evolve(nsteps=5, evolve_time=5)
    corr_real = ac.auto_corr.real
    exact_real = get_exact_autocorr(mol_list1, temperature, ac.evolve_times_array).real
    atol = 1e-2
    # direct comparison may fail because of different sign
    assert np.allclose(corr_real, exact_real, atol=atol) or np.allclose(corr_real, -exact_real, atol=atol)


def get_exact_autocorr(mol_list, temperature, time_series):

    nsites = len(mol_list)
    J = mol_list.j_constant.as_au()
    ph = mol_list[0].dmrg_phs[0]
    ph_levels = ph.n_phys_dim
    omega = ph.omega[0]
    g = - ph.coupling_constant
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    qn_idx = get_qnidx(ph_levels, nsites)
    H = get_hamiltonian(nsites, J, omega, g, clist, blist).extract_states(qn_idx)
    init_state = (-temperature.to_beta() * H).expm().unit()

    terms = []
    for i in range(nsites - 1):
        terms.append(clist[i].dag() * clist[i + 1])
        terms.append(-clist[i] * clist[i + 1].dag())
    j_oper = sum(terms).extract_states(qn_idx)

    corr = qutip.correlation(H, init_state, [0], time_series, [], j_oper, j_oper)[0]
    return corr



