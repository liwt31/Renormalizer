# -*- coding: utf-8 -*-

from typing import List, Union, Dict, Callable

from renormalizer.model.basis import BasisSet, BasisSimpleElectron, BasisMultiElectronVac, BasisHalfSpin, BasisSHO
from renormalizer.model.mol import Mol, Phonon
from renormalizer.model.op import Op
from renormalizer.utils import Quantity, cached_property

from renormalizer.model import Model

class UniformModel(Model):

    def __init__(self, basis: List[BasisSet], ham_terms: List[Op], dipole: Dict = None, unitcell_num: int=None):
        super().__init__(basis, ham_terms, dipole)
        if unitcell_num is None:
            self.unitcell_num = 2
        else:
            self.unitcell_num = unitcell_num
        assert isinstance(self.unitcell_num, int)

        self.unitcell_size = len(basis) // self.unitcell_num

    def flip(self):
        return self.__class__(self.basis[::-1], self.ham_terms, self.dipole, self.unitcell_num)