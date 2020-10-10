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
from renormalizer.lib.krylov.krylov import expm_krylov
from renormalizer.utils import Quantity

class UMPS:

    @classmethod
    def random(cls, model: UniformModel, bond_dim: int):
        mps = cls(model)
        mps._mp = []
        np.random.seed(2021)
        for basis in model.basis[:model.unitcell_size]:
            mt = np.random.rand(bond_dim, basis.nbas, bond_dim) - 0.5
            mps._mp.append(mt)
        mps.normalize()
        return mps

    @classmethod
    def identity(cls, model: UniformModel, bond_dim: int):
        mps = cls(model)
        mps._mp = []
        np.random.seed(2021)
        for basis in model.basis[:model.unitcell_size]:
            """
            mt = np.zeros((bond_dim, basis.nbas, basis.nbas, bond_dim))
            mt[0, :, :, 0] = np.eye(basis.nbas)
            mt[1:, :, :, 1:] = np.random.rand(bond_dim-1, basis.nbas, basis.nbas, bond_dim-1)
            """
            mt = mps.regular_epsilon * np.random.rand(bond_dim, basis.nbas, basis.nbas, bond_dim)
            mt[0, :, :, 0] = np.eye(basis.nbas)
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
        self.regular_epsilon = 1e-10
        self._al_mp = []
        self._l_mp = []
        self._ar_mp = []
        self._r_mp = []
        # only set during time evolution
        self._c_mp = []

    @property
    def boundary_bond_dim(self):
        assert self._mp[0].shape[0] == self._mp[-1].shape[-1]
        return self._mp[0].shape[0]

    @property
    def bond_dims(self):
        return [mt.shape[0] for mt in self._mp]

    def _build_E_oper(self, mps=None):
        if mps is None:
            mps = self._mp
        # bond dimension of the last matrix
        M = self.boundary_bond_dim
        def E_oper_func(x):
            x = x.reshape(M, M)
            """
            - ms'- ms'- |
               |    |   x
            - ms - ms - |
            """
            argument_list = []
            for i, mt in enumerate(mps):
                if mt.ndim == 3:
                    # top, middle, bottom
                    args = [mt.conj(), (f"t{i}", f"m{i}", f"t{i+1}"), mt, (f"b{i}", f"m{i}", f"b{i+1}")]
                elif mt.ndim == 4:
                    # auxiliary, top, middle, bottom
                    args = [mt.conj(), (f"t{i}", f"m{i}", f"a{i}",  f"t{i+1}"), mt, (f"b{i}", f"m{i}", f"a{i}", f"b{i+1}")]
                else:
                    assert False
                argument_list.extend(args)
            argument_list.extend([x, (f"t{len(mps)}", f"b{len(mps)}"), ("t0", "b0")])
            return contract(*argument_list)

        return LinearOperator((M ** 2, M ** 2), E_oper_func)

    def _flip(self):
        flipped_mps = self._metacopy()
        flipped_mps.model = flipped_mps.model.flip()
        flipped_mps.l, flipped_mps.r = self.r, self.l
        flipped_mps._mp = [flip_mt(mt) for mt in reversed(self._mp)]
        flipped_mps._ar_mp = [flip_mt(mt) for mt in reversed(self._al_mp)]
        flipped_mps._r_mp = [flip_mt(mt) for mt in reversed(self._l_mp)]
        flipped_mps._al_mp = [flip_mt(mt) for mt in reversed(self._ar_mp)]
        flipped_mps._l_mp = [flip_mt(mt) for mt in reversed(self._r_mp)]
        flipped_mps._c_mp = [ flip_mt(mt) for mt in reversed(self._c_mp)]
        return flipped_mps

    def _build_E_oper_T(self):
        return self._build_E_oper(mps=self._flip())

    def normalize(self, enforce_identity=None):
        # This function is extremely time consuming. Avoid normalization when possible
        try:
            assert self.l is not None and self.r is not None
            self._check_normalize()
        except AssertionError:
            pass
        else:
            return self

        if self.boundary_bond_dim == 1:
            # for testing
            evecs1 = evecs2 = np.array([[1.+0j]])
            evals1 = evals2 = self._build_E_oper() @ evecs1[:, 0]
        else:
            evals1, evecs1 = eigs(self._build_E_oper(), k=1, which="LM")
            evals2, evecs2 = eigs(self._build_E_oper_T(), k=1, which="LM")
        # complex eval, evecs are usually caused by degenerate eigenvalues
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
        M = self.boundary_bond_dim
        if enforce_identity is None:
            if hash(l.tobytes()) % 2 == 1:
                r *= 1 / (l @ r)
            else:
                l *= 1 / (l @ r)
        elif enforce_identity is "l":
            l *= np.sqrt(M)
            assert np.allclose(l.reshape(M, M), np.eye(M))
            r *= 1 / (l @ r)
        else:
            assert enforce_identity is "r"
            r *= np.sqrt(M)
            assert np.allclose(r.reshape(M, M), np.eye(M))
            l *= 1 / (l @ r)
        self.l = l.reshape(M, M)
        self.r = r.reshape(M, M)
        self._check_normalize()
        return self

    def _check_normalize(self):
        assert self.r.shape == self.l.shape == (self.boundary_bond_dim, self.boundary_bond_dim)
        assert np.allclose(self._build_E_oper() @ self.r.ravel(), self.r.ravel())
        assert np.allclose(self._build_E_oper_T() @ self.l.ravel(), self.l.ravel())
        assert abs(self.l.ravel() @ self.r.ravel() - 1) < 1e-10

    def _check_left_canonicalise(self):
        assert len(self._al_mp) == len(self._l_mp) == len(self._mp)
        assert self.l == self.r == None
        for mt in self._al_mp:
            if mt.ndim == 3:
                assert np.allclose(contract("abc, abd -> cd", mt.conj(), mt), np.eye(mt.shape[-1]))
            else:
                assert mt.ndim == 4
                assert np.allclose(contract("abcd, abcf -> df", mt.conj(), mt), np.eye(mt.shape[-1]))
        for i in range(len(self._mp)):
            a1 = self._mp[i]
            l1 = self._l_mp[i]
            l2 = self._l_mp[(i+1)%len(self._mp)]
            contracted = contract_sandwitch(l1, a1, np.linalg.pinv(l2))
            assert np.allclose(contracted, self._al_mp[i]) or np.allclose(contracted, -self._al_mp[i])

    def _check_right_canonicalise(self):
        self._flip()._check_left_canonicalise()

    def left_canonicalise(self):
        # not useful at all. Delete just in case
        self.l = self.r = None
        # modify al_mp and l_mp inplace
        self._al_mp = [None] * len(self._mp)
        self._l_mp = [np.random.rand(dim, dim) for dim in self.bond_dims]
        while True:
            old_l_mp = self._l_mp.copy()
            for i in range(len(self._mp)):
                next_i = (i + 1) % len(self._mp)
                # LA[i] = $L[i] \times A[i] $
                lai = np.tensordot(self._l_mp[i], self._mp[i], axes=1)
                # $A_L[i]$, $L[i+1]$
                ali, lip1 = scipy.linalg.qr(lai.reshape(-1, self.bond_dims[next_i]), mode="economic")
                self._al_mp[i] = ali.reshape(self._mp[i].shape)
                self._l_mp[next_i] = lip1
            # all matrices are converged. Exit.
            for old, new in zip(old_l_mp, self._l_mp):
                if (not np.allclose(old, new, rtol=1e-9, atol=1e-12)) and (not np.allclose(old, -new, rtol=1e-9, atol=1e-12)):
                    break
            else:
                break
        self._check_left_canonicalise()
        return self

    def right_canonicalise(self):
        al_mp_backup = self._al_mp.copy()
        l_mp_backup = self._l_mp.copy()
        mps = self._flip().left_canonicalise()
        self._ar_mp = [flip_mt(mt) for mt in reversed(mps._al_mp)]
        self._r_mp = [flip_mt(mt) for mt in reversed(mps._l_mp)]
        self._al_mp = al_mp_backup
        self._l_mp = l_mp_backup
        self._check_right_canonicalise()

    def compress(self, bond_dim):
        # todo: test 3 sites or more?
        mps = self.copy()
        mps.left_canonicalise()
        mps.right_canonicalise()
        c_list = [mps._l_mp[i] @ mps._r_mp[i] for i in range(len(mps._mp))]
        for i in range(len(mps._mp)):
            c = c_list[i]
            u, s, vt = scipy.linalg.svd(c)
            u = u[:, :bond_dim]
            vt = vt[:bond_dim, :]
            mps._al_mp[i-1] = np.tensordot(mps._al_mp[i-1], u, axes=1)
            mps._al_mp[i] = np.tensordot(u.T.conj(), mps._al_mp[i], axes=1)
            mps._l_mp[i] = u.T.conj() @ mps._l_mp[i]
            mps._ar_mp[i] = np.tensordot(vt, mps._ar_mp[i], axes=1)
            mps._ar_mp[i-1] = np.tensordot(mps._ar_mp[i-1], vt.T.conj(), axes=1)
            mps._r_mp[i-1] = mps._r_mp[i-1] @ vt.T.conj()

        mps._mp = mps._al_mp
        mps.normalize()
        return mps

    def _repeat_mps(self, mpo):
        assert len(mpo) % len(self) == 0
        mps = self._metacopy()
        mps._mp = self._mp * (len(mpo) // len(self))
        mps.l, mps.r = self.l, self.r
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
        self.normalize()
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

    def my_pinv(self, S):
        u, s, vt = np.linalg.svd(S)
        new_s = s + self.regular_epsilon
        s_inv = np.diag(1 / new_s)
        return vt.T.conj() @ s_inv @ u.T.conj()

    def calc_deriv(self, mpo):
        self._check_normalize()
        e = self.expectation(mpo)
        del mpo
        h_mpo = Mpo(self.model, offset=Quantity(e))
        L_h = self._solve_L_h(h_mpo)
        R_h = self._solve_R_h(h_mpo)
        # todo: reduce number of sites. Not that this does not affect calculation in `Environ`
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
            """
            | a      d|
            l         r
            |b   |c  e|
             --- ms---  
            """
            F1 = contract_sandwitch(left_part, mt, right_part)
            # The second term
            left_part = env_h.read("L", idx - 1)
            right_part = env_h.read("R", idx + 1)
            """
             a       f
             --    --- 
            |  b |d g |
            l -- o -- r
            |c   |e  h|
             --- ms---  
            """
            if mt.ndim == 3:
                F2 = contract("abc, ceh, bdeg, fgh -> adf", left_part, mt, h_mpo[idx], right_part)
            else:
                assert mt.ndim == 4
                F2 = contract("abc, ceih, bdeg, fgh -> adif", left_part, mt, h_mpo[idx], right_part)
            # The third term
            left_part = env_h.read("L", idx + nsite - 1)
            right_part = env_h.read("R", idx + nsite + 1)
            if mt.ndim == 3:
                F3 = contract("abc, ceh, bdeg, fgh -> adf", left_part, mt, h_mpo[idx + nsite], right_part)
            else:
                assert mt.ndim == 4
                F3 = contract("abc, ceih, bdeg, fgh -> adif", left_part, mt, h_mpo[idx + nsite], right_part)
            # the fourth term
            left_part = env.read("L", idx - 1).squeeze(axis=1)
            right_part = env_rh.read("R", idx + 1).squeeze(axis=1)
            F4 = contract_sandwitch(left_part, mt, right_part)
            F = F1 + F2 + F3 + F4
            if mt.ndim == 4:
                F = F.reshape(F.shape[0], d**2, F.shape[-1])
            l_half = scipy.linalg.sqrtm(env.read("L", idx - 1).squeeze(axis=1))
            V_L_target = np.tensordot(l_half, mt.conj(), axes=1).reshape(-1, M).T
            V_L = scipy.linalg.null_space(V_L_target)
            assert np.allclose(V_L.conj().T @ V_L, np.eye(V_L.shape[-1]))
            if mt.ndim == 3:
                V_L = V_L.reshape(M, d, -1)
            else:
                assert mt.ndim == 4
                V_L = V_L.reshape(M, d**2, -1)
            l_half_inv = self.my_pinv(l_half)
            r_inv = self.my_pinv(env.read("R", idx + 1).squeeze(axis=1))
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
        #sol = solve_ivp(fun, (0, dt), init_y, method="RK45", rtol=1e-5, atol=1e-8)
        return self.__class__.from_array(sol.y[:, -1], self)

    def evolve_mixed(self, mpo, dt):
        try:
            self._check_left_canonicalise()
            self._check_right_canonicalise()
        except AssertionError:
            self.left_canonicalise()
            self.right_canonicalise()
        nsite = len(self)
        #if not self._c_mp:
        self._c_mp = [self._l_mp[i] @ self._r_mp[(i+1)%nsite] for i in range(nsite)]
        new_c_mp = [None] * len(self)
        new_ac_mp = [None] * len(self)
        l_umps = self._metacopy()
        l_umps._mp = self._al_mp
        l_umps.normalize(enforce_identity="l")
        e = l_umps.expectation(mpo)
        del mpo
        h_mpo = Mpo(self.model, offset=Quantity(e))
        L_h = l_umps._solve_L_h(h_mpo)
        r_umps = self._metacopy()
        r_umps._mp = self._ar_mp
        r_umps.normalize(enforce_identity="r")
        R_h = r_umps._solve_R_h(h_mpo)

        # The environment has already been computed when calculating expectations
        # Hopefully this is not a huge problem
        # todo: reduce the number of sites of the identity operator
        iden_mpo = Mpo.identity(self.model)
        M = self.boundary_bond_dim
        env_h_lpart = Environ(l_umps._repeat_mps(h_mpo), h_mpo, domain="L", left_sentinal=l_umps.l.reshape(M, 1, M))
        env_h_rpart = Environ(r_umps._repeat_mps(h_mpo), h_mpo, domain="R", right_sentinal=r_umps.r.reshape(M, 1, M))
        env_lh = Environ(l_umps, iden_mpo, domain="L", left_sentinal=L_h.reshape(M, 1, M))
        env_rh = Environ(r_umps, iden_mpo, domain="R", right_sentinal=R_h.reshape(M, 1, M))
        del l_umps, r_umps, e, iden_mpo
        for i in range(len(self._mp)):
            c_shape = self._c_mp[i].shape
            ac = np.tensordot(self._c_mp[i], self._ar_mp[i], axes=1)
            ac_shape = ac.shape
            assert np.allclose(ac, np.tensordot(self._al_mp[i], self._c_mp[(i+1)%nsite], axes=1)) \
                   or np.allclose(-ac, np.tensordot(self._al_mp[i], self._c_mp[(i+1)%nsite], axes=1))

            def A_C_func(A_C):
                A_C = A_C.reshape(ac_shape)
                part1 = np.tensordot(env_lh.read("L", i-1).squeeze(axis=1), A_C, axes=1)
                """
                 a       f
                 --    --- 
                |  b |d g |
                l -- o -- r
                |c   |e  h|
                 --- ms---  
                """
                if A_C.ndim == 3:
                    path = "abc, ceh, bdeg, fgh -> adf"
                else:
                    assert A_C.ndim == 4
                    path = "abc, ceih, bdeg, fgh -> adif"
                part2 = contract(path, env_h_lpart.read("L", i - 1), A_C, h_mpo[i], env_h_rpart.read("R", i + 1))
                part3 = contract(path, env_h_lpart.read("L", i + nsite - 1), A_C, h_mpo[i + nsite], env_h_rpart.read("R", i + nsite + 1))
                part4 = np.tensordot(A_C, env_rh.read("R", i+1).T.squeeze(axis=1), axes=1)
                HA_C =  part1 + part2 + part3 + part4
                return HA_C.ravel()

            ac, j = expm_krylov(A_C_func, dt, ac.ravel())
            #ac = ac
            ac /= np.linalg.norm(ac)
            ac = ac.reshape(ac_shape)

            def C_func(C):
                C = C.reshape(c_shape)
                part1 = np.tensordot(C, env_rh.read("R", i).T.squeeze(axis=1), axes=1)
                """
                |a   b   e|
                l---------r
                |c       d|
                 --- ms---  
                """
                part2 = contract("abc, cd, ebd -> ae", env_h_lpart.read("L", i-1), C, env_h_rpart.read("R", i))
                part3 = contract("abc, cd ,ebd -> ae", env_h_lpart.read("L", nsite + i -1), C, env_h_rpart.read("R", nsite+i))
                part4 = np.tensordot(env_lh.read("L", i-1).squeeze(axis=1), C, axes=1)
                HC = part1 + part2 + part3 + part4
                return HC.ravel()

            c, j = expm_krylov(C_func, dt, self._c_mp[i].ravel())
            #c = self._c_mp[i].ravel()
            c /= np.linalg.norm(c)
            c = c.reshape(c_shape)

            new_ac_mp[i] = ac
            new_c_mp[i] = c

        for i in range(nsite):
            self._al_mp[i] = update_canonical_L(new_ac_mp[i], new_c_mp[(i+1)%nsite])
            self._ar_mp[i] = update_canonical_R(new_ac_mp[i], new_c_mp[i])
        self._c_mp = new_c_mp
        # previous values are outdated
        self._mp = self._al_mp
        self.l = self.r = None
        self._l_mp = self._r_mp = []
        return self

    def conj(self):
        mps = self._metacopy()
        mps._mp = [mt.conj() for mt in self._mp]
        # todo: other properties
        # l and r should be real
        mps.l, mps.r = self.l, self.r
        return mps

    def _metacopy(self):
        new_mps = UMPS(self.model)
        return new_mps

    def copy(self):
        mps = self._metacopy()
        mps.__dict__ = self.__dict__.copy()
        return mps

    def append(self, item):
        self._mp.append(item)

    def __len__(self):
        return len(self._mp)

    def __iter__(self):
        return iter(self._mp)

    def __getitem__(self, item):
        return self._mp[item]

    def __add__(self, other):
        new_mps = self._metacopy()
        for mt1, mt2 in zip(self, other):
            shape = []
            shape.append(mt1.shape[0] + mt2.shape[0])
            assert mt1.shape[1:-1] == mt2.shape[1:-1]
            shape.extend(mt1.shape[1:-1])
            shape.append(mt1.shape[-1] + mt2.shape[-1])
            new_mt = np.zeros(shape, mt1.dtype)
            new_mt[:mt1.shape[0], ..., :mt1.shape[0]] = mt1
            new_mt[-mt2.shape[0]:, ..., -mt2.shape[0]:] = mt2
            new_mps.append(new_mt)
        if None not in [self.l, self.r, other.l, other.r]:
            # todo: there is an analytical way
            # new_mps.l = np.concatenate([self.l, other.l])
            # new_mps.r = np.concatenate([self.r, other.r])
            # new_mps._check_normalize()
            pass
        return new_mps



def flip_mt(mt: np.ndarray):
    if mt.ndim == 2:
        return mt.transpose()
    elif mt.ndim == 3:
        return mt.transpose(2, 1, 0)
    else:
        assert mt.ndim == 4
        return mt.transpose(3, 1, 2, 0)


def contract_sandwitch(left, mt, right):
    if mt.ndim == 3:
        return contract("ab, bcd, de -> ace", left, mt, right)
    else:
        assert mt.ndim == 4
        return contract("ab, bcfd, de -> acfe", left, mt, right)


def update_canonical_L(A_C, C):
    # P30
    ac_shape = A_C.shape
    U_AC, P_AC = scipy.linalg.polar(A_C.reshape(-1, ac_shape[-1]))
    U_C, P_C = scipy.linalg.polar(C)
    A_L = U_AC @ U_C.T.conj()
    A_L = A_L.reshape(ac_shape)
    print(np.linalg.norm(np.tensordot(A_L, C, axes=1) - A_C), np.linalg.norm(P_AC - P_C))
    return A_L


def update_canonical_R(A_C, C):
    # P30
    ac_shape = A_C.shape
    U_AC, P_AC = scipy.linalg.polar(A_C.reshape(ac_shape[0], -1), "left")
    U_C, P_C = scipy.linalg.polar(C, "left")
    A_R = U_C.T.conj() @ U_AC
    A_R = A_R.reshape(ac_shape)
    print(np.linalg.norm(np.tensordot(C, A_R, axes=1) - A_C), np.linalg.norm(P_AC - P_C))
    return A_R