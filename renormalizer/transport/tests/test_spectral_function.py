# -*- coding: utf-8 -*-

import numpy as np

import qutip
import pytest

from renormalizer.model import Phonon, Mol, HolsteinModel, Model, Op
from renormalizer.model.basis import BasisSimpleElectron, BasisSHO
from renormalizer.transport.spectral_function import SpectralFunction, SpectralFunctionZT
from renormalizer.utils import Quantity
from renormalizer.utils.configs import CompressConfig, EvolveMethod, EvolveConfig, CompressCriteria
from renormalizer.utils.qutip_utils import get_clist, get_blist, get_holstein_hamiltonian


@pytest.mark.parametrize("temperature", (0, 1))
def test_spectral_function_holstein(temperature):
    nsites = 3
    nlevels = 2
    ph = Phonon.simple_phonon(Quantity(1), Quantity(1), nlevels)
    mol = Mol(Quantity(0), [ph])
    model = HolsteinModel([mol] * nsites, Quantity(1), periodic=True)
    temperature = Quantity(temperature)
    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    if temperature == 0:
        sf = SpectralFunctionZT(model, compress_config=compress_config, evolve_config=evolve_config)
    else:
        sf = SpectralFunction(model, temperature, compress_config=compress_config, evolve_config=evolve_config)
    sf.evolve(nsteps=10, evolve_time=5)
    qutip_res = get_qutip_holstein_sf(nsites, 1, nlevels, 1, np.sqrt(1/2), temperature, sf.evolve_times_array)
    rtol = 1e-4
    assert np.allclose(sf.G_array[:, 1], qutip_res, rtol=rtol)


def test_spectral_function_bogoliubov():
    temperature = Quantity(1)
    nsites = 2
    omega = 1
    # must be large enough for the algorithm to work
    nlevels = 16
    basis = []
    g = 1

    compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=24)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)

    ph = Phonon.simple_phonon(Quantity(omega), Quantity(g/np.sqrt(omega/2)), nlevels)
    mol = Mol(Quantity(0), [ph])
    model = HolsteinModel([mol] * nsites, Quantity(1))
    sf1 = SpectralFunction(model, temperature, compress_config=compress_config.copy(), evolve_config=evolve_config.copy())
    #sf1 = SpectralFunctionZT(model, compress_config=compress_config.copy(), evolve_config=evolve_config.copy())
    sf1.evolve(nsteps=5, evolve_time=2.5)

    # The bogoliubov model
    for i in range(nsites):
        basis.append(BasisSimpleElectron(i))
        basis.append(BasisSHO((i, 0), omega, nlevels))
        basis.append(BasisSHO((i, 1), omega, nlevels))

    ham_terms = []
    for i in range(nsites-1):
        hop1 = Op(r"a^\dagger a", [i, i+1])
        hop2 = Op(r"a^\dagger a", [i+1, i])
        ham_terms.extend([hop1, hop2])

    for i in range(nsites):
        e0 = Op(r"a^\dagger a", i) * g ** 2 * omega
        # ph energy
        e1 = Op(r"b^\dagger b", (i, 0), omega)
        e2 = Op(r"b^\dagger b", (i, 1), -omega)
        # coupling
        theta = np.arctanh(np.exp(-temperature.to_beta() * omega / 2))
        coup1 = - g * np.cosh(theta) * omega * Op(r"a^\dagger a", i) * Op(r"b^\dagger + b", (i, 0))
        coup2 = - g * np.sinh(theta) * omega * Op(r"a^\dagger a", i) * Op(r"b^\dagger + b", (i, 1))
        ham_terms.extend([e0, e1, e2, coup1, coup2])

    model = Model(basis, ham_terms)

    sf2 = SpectralFunctionZT(model, compress_config=compress_config, evolve_config=evolve_config)
    sf2.evolve(nsteps=5, evolve_time=2.5)
    assert np.allclose(sf1.G_array, sf2.G_array, rtol=1e-3)


def get_qutip_holstein_sf(nsites, J, ph_levels, omega, g, temperature, time_series):
    if temperature == 0:
        beta = 1e100
    else:
        beta = temperature.to_beta()
    clist = get_clist(nsites, ph_levels)
    blist = get_blist(nsites, ph_levels)

    H = get_holstein_hamiltonian(nsites, J, omega, g, clist, blist, True)
    init_state_list = []
    for i in range(nsites):
        egs = qutip.basis(2, 0)
        init_state_list.append(egs * egs.dag())
        b = qutip.destroy(ph_levels)
        init_state_list.append((-beta * (omega * b.dag() * b)).expm().unit())
    init_state = qutip.tensor(init_state_list)

    return qutip.correlation(H, init_state, [0], time_series, [], clist[1], clist[0].dag())[0] / 1j
