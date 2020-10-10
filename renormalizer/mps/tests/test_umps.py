# -*- coding: utf-8 -*-

import logging
import math

import pytest
from opt_einsum import contract

from renormalizer.model import UniformModel, Op
from renormalizer.model.basis import BasisHalfSpin, BasisSimpleElectron, BasisSHO
from renormalizer.mps.umps import UMPS
from renormalizer.mps import Mpo
from renormalizer.mps.backend import np
from renormalizer.utils.log import set_stream_level


def test_ising():
    Gamma = J = 1
    basis = [BasisHalfSpin(0), BasisHalfSpin(1)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_x S_x", [0, 1], -J)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.random(model, 16)
    mpo = Mpo(model)
    print(mps.expectation(mpo))
    for i in range(5):
        mps = mps.evolve(mpo, -0.4)
        print(mps.expectation(mpo))


def test_ising_2site():
    Gamma = J = 1
    basis = [BasisHalfSpin(0), BasisHalfSpin(1), BasisHalfSpin(2), BasisHalfSpin(3)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_z", 1, -Gamma), Op(r"S_x S_x", [0, 1], -J), Op(r"S_x S_x", [1, 2], -J)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.random(model, 16)
    mpo = Mpo(model)
    print(mps.expectation(mpo) / 2)
    for i in range(5):
        mps = mps.evolve(mpo, -0.4)
        print(mps.expectation(mpo)/2)


def test_harmonic_oscillator():
    omega = 1
    beta = 1
    basis = [BasisSHO(0, omega, 20), BasisSHO(1, omega, 20)]
    ham_terms = [Op(r"b^\dagger b", 0, omega)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.identity(model, 2)
    mpo = Mpo(model)
    mps = mps.evolve_mixed(mpo, -beta/2)
    assert pytest.approx(1/(math.e-1), mps.expectation(Mpo(model, Op(r"b^\dagger b", 0))))


def hueckel():
    set_stream_level(logging.INFO)
    J = 1
    epsilon = 1
    basis = [BasisSimpleElectron("e0"), BasisSimpleElectron("e1")]
    ham_terms = [Op(r"a^\dagger a", ["e0", "e1"], -J), Op(r"a^\dagger a", ["e1", "e0"], -J), Op(r"a^\dagger a", "e0", epsilon)]
    #ham_terms = [Op(r"a^\dagger a", "e0", epsilon)]
    model = UniformModel(basis, ham_terms)
    M = 5
    mps = UMPS.identity(model, M)
    mpo = Mpo(model)
    print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("a^\dagger a", "e0"))))
    # no analytical solution. Only DOS * F-D-Distribution numerical integration
    for i in range(10):
        mps = mps.evolve_mixed(mpo, -0.05)
        print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("a^\dagger a", "e0"))))

    vl = np.random.rand(1, 5)
    vr = np.random.rand(5, 1)
    print(vl @ mps.r.reshape(M, M) @ vl.T, vr.T @ mps.l.reshape(M, M) @ vr)
    E = mps._build_E_oper()
    res = (vr @ vr.T).ravel()
    new_res = E @ res
    while not np.allclose(res, new_res):
        new_res, res = E @ new_res, new_res
    print((vl.T @ vl).ravel() @ res)
    J = 10
    ad = np.array([[0, 1], [0, 0]])
    a = ad.T
    iden = np.eye(2)
    zero = np.zeros((2, 2))
    j_oper = np.array([[iden, zero, zero, zero],
                       [J * a, zero, zero, zero],
                       [-J * ad, zero, zero, zero],
                       [zero, ad, a, iden]]).transpose(0, 2, 3, 1)
    new_mps = mps._metacopy()
    new_mps._mp = [contract("abcd, ecfg-> aebfdg", j_oper, mps[0]).reshape(20, 2, 2, 20)]
   # new_mps.normalize()
    ol = np.array([0, 0, 0, 1]).reshape(1, 4)
    o_r = np.array([1, 0, 0, 0]).reshape(4, 1)
    outer_l = (vl.T @ ol).ravel()
    outer_r = (vr @ o_r.T).ravel()
    E = new_mps._build_E_oper()
    res = (outer_r.reshape(-1, 1) @ outer_r.reshape(1, -1)).ravel()
    new_res = E @ res
    while not np.allclose(res, new_res):
        new_res, res = E @ new_res, new_res
    print((outer_l.reshape(-1, 1) @ outer_l.reshape(1, -1)).ravel() @ res)
    pass
    #print(outer_l @ new_mps.l.reshape(M * 4, M * 4) @ outer_l, outer_r @ new_mps.r.reshape(M * 4, M * 4) @ outer_r)

def holstein():
    import time
    set_stream_level(logging.INFO)
    J = 1
    epsilon = 1
    lam = 1
    omega = 0.2
    nbas = 40
    basis = [BasisSimpleElectron("e0"), BasisSHO("ph0", omega, nbas), BasisSimpleElectron("e1"), BasisSHO("ph1", omega, nbas)]
    ham_terms = [Op(r"a^\dagger a", ["e0", "e1"], -J), Op(r"a^\dagger a", ["e1", "e0"], -J), Op(r"a^\dagger a", "e0", epsilon)]
    ham_terms.append(Op(r"b^\dagger b", "ph0", omega))
    ham_terms.append(Op(r"a^\dagger a", "e0") * Op(r"b^\dagger + b", "ph0", -lam))
    model = UniformModel(basis, ham_terms)
    mps = UMPS.identity(model, 64)
    mpo = Mpo(model)
    print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("b^\dagger b", "ph0"))))
    for i in range(10):
        t = time.time()
        mps = mps.evolve_mixed(mpo, -0.05)
        print(time.time() - t)
        print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("a^\dagger a", "e0"))), mps.expectation(Mpo(model, Op("b^\dagger b", "ph0"))))

def holstein_finite():
    import time
    set_stream_level(logging.INFO)
    J = 1
    epsilon = 1
    lam = 1
    omega = 0.2
    nbas = 40
    from renormalizer.model import Phonon, Mol, HolsteinModel
    from renormalizer.utils import Quantity, EvolveMethod
    from renormalizer.mps import MpDm
    ph = Phonon.simplest_phonon(Quantity(omega), Quantity(lam), lam=True, max_pdim=nbas)
    mol = Mol(Quantity(epsilon), [ph])
    model = HolsteinModel([mol]*10, Quantity(J), periodic=True)
    mpo = Mpo(model)
    mps = MpDm.max_entangled_ex(model).expand_bond_dimension(mpo)
    mps.evolve_config.method = EvolveMethod.tdvp_ps
    print(mps.expectation(mpo),mps.e_occupations, mps.ph_occupations)
    for i in range(10):
        t = time.time()
        mps = mps.evolve(mpo, -0.05*1j)
        print(time.time() - t)
        print(mps.expectation(mpo), mps.e_occupations, mps.ph_occupations)

def test_canonicalise_1site():
    Gamma = J = 1
    basis = [BasisHalfSpin(0), BasisHalfSpin(1)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_x S_x", [0, 1], -J)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.random(model, 16)
    mpo = Mpo(model)
    print(mps.expectation(mpo))
    mps.left_canonicalise()
    cano_mps = mps._metacopy()
    cano_mps._mp = mps._al_mp
    print(cano_mps.expectation(mpo))
    mps.right_canonicalise()
    cano_mps._mp = mps._ar_mp
    print(cano_mps.expectation(mpo))
    mps._flip()._check_left_canonicalise()
    compress_mps = mps.compress(15)
    print(compress_mps.expectation(mpo))

def test_canonicalise_2site():
    Gamma = J = 1
    basis = [BasisHalfSpin(0), BasisHalfSpin(1), BasisHalfSpin(2), BasisHalfSpin(3)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_z", 1, -Gamma), Op(r"S_x S_x", [0, 1], -J), Op(r"S_x S_x", [1, 2], -J)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.random(model, 16)
    mps._mp[1] = mps._mp[0]
    mpo = Mpo(model)
    print(mps.expectation(mpo))
    mps.left_canonicalise()
    cano_mps = mps._metacopy()
    cano_mps._mp = mps._al_mp
    print(cano_mps.expectation(mpo))
    mps.right_canonicalise()
    cano_mps._mp = mps._ar_mp
    print(cano_mps.expectation(mpo))
    mps._flip()._check_left_canonicalise()
    compress_mps = mps.compress(15)
    print(compress_mps.expectation(mpo))


def test_ising_PC():
    Gamma = J = 1
    # Pauli mat
    # Pauli mat
    sx = np.array([[0, 0.5], [0.5, 0]])
    sz = np.array([[0.5, 0], [0, -0.5]])
    iden = np.array([[1, 0], [0, 1]])
    zero = np.array([[0, 0], [0, 0]])

    mo = np.array([[iden, zero, zero],
                   [sx, zero, zero],
                   [sz, sx, iden]]
                  ).transpose(0, 2, 3, 1)

    basis = [BasisHalfSpin(0), BasisHalfSpin(1)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_x S_x", [0, 1], -J)]
    model = UniformModel(basis, ham_terms)
    bond_dim = 16
    mps = UMPS.random(model, bond_dim)
    mpo = Mpo(model)
    print(mps.expectation(mpo))
    dt = -0.01
    for i in range(100):
        deriv_mps = mps._metacopy()
        deriv_mps._mp = [dt * contract("abc, debf -> daefc", mps._mp[0], mo).reshape(bond_dim*3, 2, bond_dim*3)]

        mps = deriv_mps + mps
        mps.left_canonicalise()
        mps.right_canonicalise()
        mps.compress(bond_dim)
        print(mps.expectation(mpo))


def test_ising_mixed():
    Gamma = J = 1
    basis = [BasisHalfSpin(0), BasisHalfSpin(1)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_x S_x", [0, 1], -J)]
    basis = [BasisHalfSpin(0), BasisHalfSpin(1), BasisHalfSpin(2), BasisHalfSpin(3)]
    ham_terms = [Op(r"S_z", 0, -Gamma), Op(r"S_z", 1, -Gamma), Op(r"S_x S_x", [0, 1], -J), Op(r"S_x S_x", [1, 2], -J)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.random(model, 16)
    mpo = Mpo(model)
    print(mps.expectation(mpo))
    for i in range(100):
        mps = mps.evolve_mixed(mpo, -0.05)
        print(mps.expectation(mpo))


holstein_finite()