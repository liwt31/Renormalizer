# -*- coding: utf-8 -*-

from typing import List

from scipy.sparse.linalg import LinearOperator, eigs, bicgstab
import scipy.linalg
from opt_einsum import contract

from renormalizer.mps.backend import np
from renormalizer.model.uniform_model import UniformModel
from renormalizer.mps.lib import Environ
from renormalizer.mps.mpo import Mpo
from renormalizer.lib.integrate.integrate import solve_ivp
from renormalizer.utils import Quantity

class UMPS:

    @classmethod
    def random(cls, model: UniformModel, bond_dim: int):
        mps = cls(model)
        mps._mp = []
        np.random.seed(2021)
        for basis in model.basis[:model.unitcell_size]:
            mps._mp.append(np.random.rand(bond_dim, basis.nbas, bond_dim) - 0.5)
        mps.normalize()
        return mps

    @classmethod
    def random_with_qn(cls, model: UniformModel, bond_dim: int):
        # qntot is assumed to be 1. QN is half 0, half 1
        qn = np.ones(bond_dim)
        qn[:bond_dim//2] = 0
        mps = cls(model)
        mps._mp = []
        np.random.seed(2021)
        for basis in model.basis[:model.unitcell_size]:
            mt = np.random.rand(bond_dim, basis.nbas, bond_dim) - 0.5
            zero_indices = np.subtract.outer(np.add.outer(qn, basis.sigmaqn), qn) != 0
            mt[zero_indices] = 0
            mps._mp.append(mt)
        mps.normalize()
        return mps

    @classmethod
    def from_array(cls, array, ref_mps: "UMPS"):
        mps = cls(ref_mps.model)
        start = 0
        for mt in ref_mps:
            mps.append(array[start:start+mt.size].reshape(mt.shape))
            start += mt.size
        mps.normalize()
        return mps

    def __init__(self, model: UniformModel):
        self._mp = []
        self.model: UniformModel = model
        self.l = None
        self.r = None

    @property
    def boundary_bond_dim(self):
        assert self._mp[0].shape[0] == self._mp[-1].shape[-1]
        return self._mp[0].shape[0]

    def _build_E_oper(self, mps=None):
        if mps is None:
            mps = self._mp
        # bond dimension of the last matrix
        M = self.boundary_bond_dim
        def E_oper_func(x):
            x = x.reshape(M, M)
            """
            - ms - ms - |
               |    |   x
            - ms'- ms'- |
            """
            argument_list = []
            for i, mt in enumerate(mps):
                # top, middle, bottom
                argument_list.extend([mt, (f"t{i}", f"m{i}", f"t{i+1}"), mt.conj(), (f"b{i}", f"m{i}", f"b{i+1}")])
            argument_list.extend([x, (f"t{len(mps)}", f"b{len(mps)}"), ("t0", "b0")])
            return contract(*argument_list)

        return LinearOperator((M ** 2, M ** 2), E_oper_func)

    def _flip(self):
        flipped_mps = self._metacopy()
        flipped_mps.l, flipped_mps.r = flipped_mps.r, flipped_mps.l
        for mt in reversed(self._mp):
            flipped_mps.append(mt.transpose(2, 1, 0))
        return flipped_mps

    def _build_E_oper_T(self):
        return self._build_E_oper(mps=self._flip())

    def normalize(self):
        # This function is extremely time consuming. Avoid normalization when possible
        evals1, evecs1 = eigs(self._build_E_oper(), k=2, which="LM")
        evals2, evecs2 = eigs(self._build_E_oper_T(), k=2, which="LM")
        assert np.allclose(evals1, evals2)
        normalize_coef = np.sqrt(evals1[0] ** (1/len(self)))
        assert np.allclose(normalize_coef.imag, 0)
        normalize_coef = normalize_coef.real
        self._mp = [mt / normalize_coef for mt in self]
        r = evecs1[:, 0]
        r *= np.exp(-1j * np.angle(r.ravel()[0]))
        l = evecs2[:, 0]
        l *= np.exp(-1j * np.angle(l.ravel()[0]))
        assert np.allclose(l.imag, 0) and np.allclose(r.imag, 0)
        l = l.real
        r = r.real
        if hash(l.tobytes()) % 2 == 1:
            r *= 1 / (l @ r)
        else:
            l *= 1 / (l @ r)
        M = self.boundary_bond_dim
        self.l = l.reshape(M, M)
        self.r = r.reshape(M, M)
        self._check_normalize()
        return self

    def _check_normalize(self):
        assert np.allclose(self._build_E_oper() @ self.r.ravel(), self.r.ravel())
        assert np.allclose(self._build_E_oper_T() @ self.l.ravel(), self.l.ravel())
        assert abs(self.l.ravel() @ self.r.ravel() - 1) < 1e-10

    def _repeat_mps(self, mpo):
        assert len(mpo) % len(self) == 0
        mps = self._metacopy()
        mps._mp = self._mp * (len(mpo) // len(self))
        mps._check_normalize()
        return mps

    def _contract_all_but_r(self, mpo):
        mps = self._repeat_mps(mpo)
        M = self.boundary_bond_dim
        left_sentinal = self.l.reshape(M, 1, M)
        right_sentinal = self.r.reshape(M, 1, M)
        environ = Environ(mps, mpo, domain="L", left_sentinal=left_sentinal, right_sentinal=right_sentinal)
        return environ.read("L", len(mpo)-1)

    def expectation(self, mpo):
        self._check_normalize()
        all_but_r = self._contract_all_but_r(mpo)
        #assert np.allclose(all_but_r.imag, 0)
        res = all_but_r.ravel() @ self.r.ravel()
        assert np.allclose(res.imag, 0)
        return res.real

    def _solve_L_h(self, h_mpo):
        M = self.boundary_bond_dim
        E_oper_T = self._build_E_oper_T()
        I_minus_tilde_E_T = LinearOperator(
            (M ** 2, M ** 2),
            lambda x: x - E_oper_T.dot(x) + x @ self.r.ravel() * self.l.ravel(),
        )
        target = self._contract_all_but_r(h_mpo).ravel()
        L_h, info = bicgstab(I_minus_tilde_E_T, target, tol=1e-14)
        assert info == 0

        L_h = L_h.reshape(M, M)
        assert np.allclose(L_h, L_h.T.conj())
        return L_h

    def _solve_R_h(self, h_mpo):
        flipped_h_mpo = [mt.transpose(3, 1, 2, 0) for mt in reversed(h_mpo)]
        return self._flip()._solve_L_h(flipped_h_mpo)

    def calc_deriv(self, mpo):
        self._check_normalize()
        e = self.expectation(mpo)
        del mpo
        h_mpo = Mpo(self.model, offset=Quantity(e))
        L_h = self._solve_L_h(h_mpo)
        R_h = self._solve_R_h(h_mpo)
        # todo: reduce number of sites
        iden_mpo = Mpo.identity(self.model)
        M = self.boundary_bond_dim
        left_sentinal = self.l.reshape(M, 1, M)
        right_sentinal = self.r.reshape(M, 1, M)
        env = Environ(self, iden_mpo, left_sentinal=left_sentinal, right_sentinal=right_sentinal)
        env_h = Environ(self._repeat_mps(h_mpo), h_mpo, left_sentinal=left_sentinal, right_sentinal=right_sentinal)
        env_lh = Environ(self, iden_mpo, domain="L", left_sentinal=L_h.reshape(M, 1, M))
        env_rh = Environ(self, iden_mpo, domain="R", right_sentinal=R_h.reshape(M, 1, M))
        # otherwise EOM is more complicated
        assert self.model.unitcell_num == 2
        deriv_list = []
        for idx, mt in enumerate(self):
            d = self.model.basis[idx].nbas
            nsite = self.model.unitcell_size
            # The first term.
            left_part = env_lh.read("L", idx - 1).squeeze(axis=1)
            right_part = env.read("R", idx + 1).squeeze(axis=1)
            F1 = contract("ab, acd, de -> bce", left_part, mt, right_part)
            # The second term
            left_part = env_h.read("L", idx - 1)
            right_part = env_h.read("R", idx + 1)
            """
             a       f
             -- ms --- 
            |  b |d g |
            l -- o -- r
            |c   |e  h|
             ---   ---  
            """
            F2 = contract("abc, adf, bdeg, fgh -> ceh", left_part, mt, h_mpo[idx], right_part)
            # The third term
            left_part = env_h.read("L", idx + nsite - 1)
            right_part = env_h.read("R", idx + nsite + 1)
            F3 = contract("abc, adf, bdeg, fgh -> ceh", left_part, mt, h_mpo[idx + nsite], right_part)
            # the fourth term
            left_part = env.read("L", idx - 1).squeeze(axis=1)
            right_part = env_rh.read("R", idx + 1).squeeze(axis=1)
            F4 = contract("ab, acd, de -> bce", left_part, mt, right_part)
            F = F1 + F2 + F3 + F4
            l_half = scipy.linalg.sqrtm(env.read("L", idx - 1).squeeze(axis=1))
            V_L_target = np.tensordot(l_half, mt.conj(), axes=1).reshape(-1, M).T
            V_L = scipy.linalg.null_space(V_L_target)
            # null space could be larger than expected because of the sparsity enforced by symmetry
            assert np.allclose(V_L.conj().T @ V_L, np.eye(V_L.shape[-1]))
            V_L = V_L.reshape(M, d, -1)
            l_half_inv = scipy.linalg.pinv(l_half)
            r_inv = scipy.linalg.pinv(env.read("R", idx + 1).squeeze(axis=1))
            deriv = contract(
                "ab, ce, bdh, efh, adg, gi -> cfi",
                l_half_inv,
                l_half_inv,
                V_L.conj(),
                V_L,
                F,
                r_inv,
            )
            assert np.allclose(deriv.imag, 0)
            deriv = deriv.real
            deriv_list.append(deriv.ravel())
        return np.concatenate(deriv_list)

    def evolve(self, mpo, dt):
        def fun(t, y):
            mps = self.__class__.from_array(y, self)
            return mps.calc_deriv(mpo)
        self._check_normalize()
        init_y = np.concatenate([mt.ravel() for mt in self._mp])
        sol = solve_ivp(fun, (0, dt), init_y, method="RK45")
        return self.__class__.from_array(sol.y[:, -1], self)


    def conj(self):
        mps = self._metacopy()
        mps._mp = [mt.conj() for mt in self._mp]
        return mps

    def _metacopy(self):
        new_mps = UMPS(self.model)
        new_mps.l = self.l
        new_mps.r = self.r
        return new_mps

    def append(self, item):
        self._mp.append(item)

    def __len__(self):
        return len(self._mp)

    def __iter__(self):
        return iter(self._mp)

    def __getitem__(self, item):
        return self._mp[item]