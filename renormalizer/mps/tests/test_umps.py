# -*- coding: utf-8 -*-

import logging
import math

import pytest

from renormalizer.model import UniformModel, Op
from renormalizer.model.basis import BasisHalfSpin, BasisSimpleElectron, BasisSHO
from renormalizer.mps.umps import UMPS
from renormalizer.mps import Mpo
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
    mps = mps.evolve(mpo, -beta/2)
    assert pytest.approx(1/(math.e-1), mps.expectation(Mpo(model, Op(r"b^\dagger b", 0))))


def hueckel():
    set_stream_level(logging.INFO)
    J = 1
    epsilon = 1
    basis = [BasisSimpleElectron("e0"), BasisSimpleElectron("e1")]
    ham_terms = [Op(r"a^\dagger a", ["e0", "e1"], -J), Op(r"a^\dagger a", ["e1", "e0"], -J), Op(r"a^\dagger a", "e0", epsilon)]
    #ham_terms = [Op(r"a^\dagger a", "e0", epsilon)]
    model = UniformModel(basis, ham_terms)
    mps = UMPS.identity(model, 1)
    mpo = Mpo(model)
    print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("a^\dagger a", "e0"))))
    for i in range(10):
        mps = mps.evolve(mpo, -0.1)
        print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("a^\dagger a", "e0"))))

def holstein():
    set_stream_level(logging.INFO)
    J = 1
    epsilon = 1
    lam = 1
    omega = 1
    nbas = 2
    basis = [BasisSimpleElectron("e0"), BasisSHO("ph0", omega, nbas), BasisSimpleElectron("e1"), BasisSHO("ph1", omega, nbas)]
    ham_terms = [Op(r"a^\dagger a", ["e0", "e1"], -J), Op(r"a^\dagger a", ["e1", "e0"], -J), Op(r"a^\dagger a", "e0", epsilon)]
    ham_terms.append(Op(r"b^\dagger b", "ph0", omega))
    ham_terms.append(Op(r"a^\dagger a", "e0") * Op(r"b^\dagger + b", "ph0", -lam))
    model = UniformModel(basis, ham_terms)
    mps = UMPS.identity(model, 4)
    mpo = Mpo(model)
    print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("b^\dagger b", "ph0"))))
    for i in range(10):
        mps = mps.evolve(mpo, -1)
        print(mps.expectation(mpo), mps.expectation(Mpo(model, Op("b^\dagger b", "ph0"))))

test_ising()