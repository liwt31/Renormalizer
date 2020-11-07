# -*- coding: utf-8 -*-

import logging

from renormalizer.mps.backend import np
from renormalizer.mps import MpDm, Mpo, ThermalProp
from renormalizer.utils import TdMpsJob, Quantity, EvolveConfig, CompressConfig, EvolveMethod
from renormalizer.model import Model


logger = logging.getLogger(__name__)


class SpectralFunction(TdMpsJob):

    def __init__(
            self,
            model: Model,
            temperature: Quantity,
            compress_config: CompressConfig = None,
            evolve_config: EvolveConfig = None,
            dump_dir: str = None,
            job_name: str=None,
    ):
        self.model: Model = model
        # no offset for simplicity. TDVP-PS is supposed to be robust to offsets
        self.h_mpo = Mpo(self.model)
        # local vibration hamiltonian in the 0 electron space.
        self._local_vib_terms = []
        for op in model.ham_terms:
            site_idx_set = set(self.model.dof_to_siteidx[dof] for dof in op.dofs)
            if not len(site_idx_set) == 1:
                # multi site operator.
                continue
            self._local_vib_terms.append(op)
            del op, site_idx_set
        self.zero_electron_model = Model(model.basis, self._local_vib_terms)
        # the mpo for exact propagation in zero electron space
        self.exact_prop_h_mpo = Mpo(self.zero_electron_model)
        self.temperature = temperature
        self.compress_config = compress_config
        if self.compress_config is None:
            self.compress_config = CompressConfig()
        # todo: a translational invariance Model
        # electron-addition Green's function at different $t$ assuming translational invariance
        self._G_array = []
        self.e_occupations_array = []
        super().__init__(evolve_config, False, dump_dir, job_name)

    @property
    def G_array(self):
        return np.array(self._G_array)

    def init_mps(self):
        i_mpo = MpDm.max_entangled_gs(self.model)
        i_mpo.compress_config = self.compress_config
        i_mpo.evolve_config = self.evolve_config
        beta = self.temperature.to_beta()
        exact_evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
        tp = ThermalProp(i_mpo, self.exact_prop_h_mpo, evolve_config=exact_evolve_config, auto_expand=False)
        tp.evolve(None, 1, beta / 2j)
        ket_mpo = tp.latest_mps
        ket_mpo.evolve_config = self.evolve_config
        creation_oper = Mpo.onsite(self.model, r"a^\dagger", dof_set={self.model.e_dofs[0]})
        a_ket_mpo = creation_oper.apply(ket_mpo, canonicalise=True)
        if self.evolve_config.is_tdvp:
            a_ket_mpo = a_ket_mpo.expand_bond_dimension(self.h_mpo)
        a_ket_mpo.canonical_normalize()
        a_bra_mpo = tp.latest_mps.copy()
        a_ket_mpo.evolve_config = self.evolve_config
        a_bra_mpo.evolve_config = exact_evolve_config
        return (a_bra_mpo, a_ket_mpo)

    def process_mps(self, mps):
        key = "a"
        if key not in self.model.mpos:
            a_opers = [Mpo.onsite(self.model, "a", dof_set={dof}) for dof in self.model.e_dofs]
            self.model.mpos[key] = a_opers
        else:
            a_opers = self.model.mpos[key]

        a_bra_mpo, a_ket_mpo = mps
        G = a_ket_mpo.expectations(a_opers, a_bra_mpo.conj()) / 1j
        self._G_array.append(G)
        self.e_occupations_array.append(a_ket_mpo.e_occupations)


    def evolve_single_step(self, evolve_dt):
        prev_bra_mpdm, prev_ket_mpdm = self.latest_mps
        latest_ket_mpdm = prev_ket_mpdm.evolve(self.h_mpo, evolve_dt)
        latest_bra_mpdm = prev_bra_mpdm.evolve(self.h_mpo, evolve_dt)
        return (latest_bra_mpdm, latest_ket_mpdm)

    def get_dump_dict(self):
        dump_dict = dict()
        dump_dict['temperature'] = self.temperature.as_au()
        dump_dict['time series'] = self.evolve_times
        dump_dict["G array"] = self.G_array
        dump_dict["electron occupations array"] = self.e_occupations_array
        return dump_dict