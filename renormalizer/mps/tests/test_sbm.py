# -*- coding: utf-8 -*-

import numpy as np
import qutip
import pytest

from renormalizer.model import Phonon, Mol, MolList
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils import Quantity, CompressConfig, EvolveConfig, basis


def get_mol():
    nphonons = 5
    ph_levels = 2

    delta = 1
    epsilon = 1

    ph_list = [Phonon.simple_phonon(Quantity(1), Quantity(1), ph_levels)] * nphonons
    m = Mol(Quantity(epsilon), ph_list, tunnel=Quantity(-delta))
    return m

def test_general_mpo_sbm():
    mol = get_mol()
    mol_list = MolList([mol], Quantity(0))
    mol_list.mol_list2_para()
    mpo = Mpo.general_mpo(mol_list, const=Quantity(-mol_list[0].gs_zpe*mol_list.mol_num))
    mpo_std = Mpo(mol_list)
    check_result(mpo, mpo_std)
    

def check_result(mpo, mpo_std):
    print("std mpo bond dims:", mpo_std.bond_dims)
    print("new mpo bond dims:", mpo.bond_dims)
    print("std mpo qn:", mpo_std.qn, mpo_std.qntot)
    print("new mpo qn:", mpo.qn, mpo_std.qntot)
    assert mpo_std.distance(mpo)/np.sqrt(mpo_std.dot(mpo_std)) == pytest.approx(0, abs=1e-5)

def test_zt():
    mol = get_mol()
    mol_list = MolList([mol], Quantity(0))

    mps = Mps.gs(mol_list, False)
    mps.compress_config = CompressConfig(threshold=1e-6)
    mps.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    mps.use_dummy_qn = True
    mpo = Mpo(mol_list)
    time_series = [0]
    spin = [1]
    for i in range(30):
        dt = mps.evolve_config.guess_dt
        mps = mps.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(1 - 2 * mps.e_occupations[0])
    exact = get_exact_zt(mol, time_series)
    assert np.allclose(exact, spin, atol=1e-3)


def test_ft():
    mol = get_mol()
    mol_list = MolList([mol], Quantity(0))
    mpo = Mpo(mol_list)
    impdm = MpDm.max_entangled_gs(mol_list)
    impdm.compress_config = CompressConfig(threshold=1e-6)
    impdm.use_dummy_qn = True
    temperature = Quantity(3)
    evolve_config = EvolveConfig(adaptive=True, guess_dt=-0.001j)
    tp = ThermalProp(impdm, mpo, evolve_config=evolve_config)
    tp.evolve(nsteps=1, evolve_time=temperature.to_beta() / 2j)
    mpdm = tp.latest_mps
    mpdm = Mpo.onsite(mol_list, r"sigma_x").contract(mpdm)
    mpdm.evolve_config = EvolveConfig(adaptive=True, guess_dt=0.1)
    time_series = [0]
    spin = [1 - 2 * mpdm.e_occupations[0]]
    for i in range(30):
        dt = mpdm.evolve_config.guess_dt
        mpdm = mpdm.evolve(mpo, evolve_dt=dt)
        time_series.append(time_series[-1] + dt)
        spin.append(1 - 2 * mpdm.e_occupations[0])
    exact = get_exact_ft(mol, temperature, time_series)
    assert np.allclose(exact, spin, atol=1e-3)


def get_exact_operator(mol):
    blist = []
    for i, ph1 in enumerate(mol.dmrg_phs):
        basis = [qutip.identity(2)]
        for j, ph2 in enumerate(mol.dmrg_phs):
            if j == i:
                state = qutip.destroy(ph1.n_phys_dim)
            else:
                state = qutip.identity(ph2.n_phys_dim)
            basis.append(state)
        blist.append(qutip.tensor(basis))

    ph_iden = [qutip.identity(ph.n_phys_dim) for ph in mol.dmrg_phs]

    sigma_x = qutip.tensor([qutip.sigmax()] + ph_iden)
    sigma_z = qutip.tensor([qutip.sigmaz()] + ph_iden)
    delta = mol.tunnel
    epsilon = mol.elocalex
    terms = [-delta * sigma_x, epsilon * sigma_z]
    for i, ph in enumerate(mol.dmrg_phs):
        g = ph.coupling_constant
        terms.append(ph.omega[0] * blist[i].dag() * blist[i])
        terms.append(ph.omega[0] * g * sigma_z * (blist[i].dag() + blist[i]))
    H = sum(terms)

    return H, sigma_x, sigma_z


def get_exact_zt(mol, time_series):
    H, _, sigma_z = get_exact_operator(mol)
    init_state = qutip.tensor([qutip.basis(2)] + [qutip.basis(ph.n_phys_dim) for ph in mol.dmrg_phs])
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]


def get_exact_ft(mol, temperature, time_series):
    H, sigma_x, sigma_z = get_exact_operator(mol)
    init_state =  sigma_x * (-temperature.to_beta() * H).expm().unit() * sigma_x.dag()
    result = qutip.mesolve(H, init_state, time_series, e_ops=[sigma_z])
    return result.expect[0]
