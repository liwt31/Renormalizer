# -*- coding: utf-8 -*-

import numpy as np

import qutip

from renormalizer.model import Phonon, Mol, HolsteinModel
from renormalizer.transport.spectral_function import SpectralFunction
from renormalizer.utils import Quantity
from renormalizer.utils.configs import CompressConfig, EvolveMethod, EvolveConfig, CompressCriteria
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_holstein_hamiltonian


def test_spectral_function_holstein():
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), 2)
    mol = Mol(Quantity(0), [ph])
    model = HolsteinModel([mol] * 5, Quantity(1))
    temperature = Quantity(50000, 'K')
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    sf = SpectralFunction(model, temperature, compress_config=compress_config, evolve_config=evolve_config)
    sf.evolve(nsteps=10, evolve_time=5)
    qutip_res = get_qutip_holstein_sf(model, temperature, sf.evolve_times_array)
    rtol = 5e-2
    assert np.allclose(sf.G_array[:, 1], qutip_res, rtol=rtol)


def get_qutip_holstein_sf(model, temperature, time_series):
    nsites = len(model)
    J = model.j_constant
    ph = model[0].ph_list[0]
    ph_levels = ph.n_phys_dim
    omega = ph.omega[0]
    g = - ph.coupling_constant
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    H = get_holstein_hamiltonian(nsites, J, omega, g, clist, blist)
    init_state_list = []
    for i in range(nsites):
        egs = qutip.basis(2, 0)
        init_state_list.append(egs * egs.dag())
        b = qutip.destroy(ph_levels)
        init_state_list.append((-temperature.to_beta() * (omega * b.dag() * b)).expm().unit())
    init_state = qutip.tensor(init_state_list)

    return qutip.correlation(H, init_state, [0], time_series, [], clist[1], clist[0].dag())[0] / 1j