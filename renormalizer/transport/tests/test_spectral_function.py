# -*- coding: utf-8 -*-

from renormalizer.model import Model, Op
from renormalizer.model.basis import BasisSimpleElectron
from renormalizer.transport.spectral_function import SpectralFunction
from renormalizer.utils import Quantity
from renormalizer.utils.configs import CompressConfig, EvolveMethod, EvolveConfig

def test_spectral_function():
    n = 21
    basis = [BasisSimpleElectron(i) for i in range(n)]
    ham_terms = []
    for i in range(n):
        next_i = (i+1) % n
        hop1 = Op(r"a^\dagger a", [i, next_i])
        hop2 = Op(r"a^\dagger a", [next_i, i])
        ham_terms.extend([hop1, hop2])

    model = Model(basis, ham_terms)
    #compress_config = CompressConfig(max_bonddim=1)
    evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    sf = SpectralFunction(model, Quantity(1), evolve_config=evolve_config)
    sf.evolve(0.01, 100)
    pass