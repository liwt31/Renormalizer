import pytest
from  typing import List

from renormalizer import Op
from renormalizer.mps.backend import np
from renormalizer.mps.tests.test_evolve import QUTIP_STEP, qutip_expectations, init_mps
from renormalizer.tests.parameter_exact import model
from renormalizer.tn import BasisTree, TensorTreeOperator, TensorTreeState
from renormalizer.tn.tree import from_mps
from renormalizer.tn.node import TreeNodeBasis
from renormalizer.utils import EvolveConfig, EvolveMethod, CompressConfig, CompressCriteria


def add_tto_offset(tts: TensorTreeState, tto: TensorTreeOperator):
    e = tts.expectation(tto)
    ham_terms = tto.ham_terms.copy()
    ham_terms.append(tts.basis.identity_op * (-e))
    return TensorTreeOperator(tto.basis, ham_terms)



def construct_tts_and_tto_chain():
    basis, tts, tto = from_mps(init_mps)
    op_n_list = [TensorTreeOperator(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    tto = add_tto_offset(tts, tto)
    return tts, tto, op_n_list


def construct_tts_and_tto_tree():
    node_list = [TreeNodeBasis([basis]) for basis in model.basis]
    # 0 - 2 - 4
    # |   |   |
    # 1   3   5
    root = node_list[2]
    root.add_child(node_list[0])
    root.add_child(node_list[3])
    root.add_child(node_list[4])
    node_list[0].add_child(node_list[1])
    node_list[4].add_child(node_list[5])
    basis = BasisTree(root)
    tto = TensorTreeOperator(basis, model.ham_terms)
    op_n_list = [TensorTreeOperator(basis, [Op(r"a^\dagger a", i)]) for i in range(3)]
    tts = TensorTreeState(basis, {0: 1})
    tto = add_tto_offset(tts, tto)
    return tts, tto, op_n_list


init_chain = construct_tts_and_tto_chain()
init_tree = construct_tts_and_tto_tree()


def check_result(tts: TensorTreeState, tto: TensorTreeOperator, time_step: float, final_time: float, op_n_list: List, atol: float=1e-4):
    expectations = [[tts.expectation(o) for o in op_n_list]]
    for i in range(round(final_time / time_step)):
        tts = tts.evolve(tto, time_step)
        es = [tts.expectation(o) for o in op_n_list]
        expectations.append(es)
    qutip_end = round(final_time / QUTIP_STEP) + 1
    qutip_interval = round(time_step / QUTIP_STEP)
    # more strict than mcd (the error criteria used for mps tests)
    np.testing.assert_allclose(expectations, qutip_expectations[:qutip_end:qutip_interval], atol=atol)


@pytest.mark.parametrize("tts_and_tto", [init_chain, init_tree])
def test_vmf(tts_and_tto):
    tts, tto, op_n_list = tts_and_tto
    # expand bond dimension
    tts = tts + tts.random(tts.basis, 1, 5).scale(1e-5, inplace=True)
    tts.canonicalise()
    tts.evolve_config = EvolveConfig(EvolveMethod.tdvp_vmf, ivp_rtol=1e-4, ivp_atol=1e-7, force_ovlp=False)
    check_result(tts, tto, 0.5, 2, op_n_list)


@pytest.mark.parametrize("tts_and_tto", [init_chain, init_tree])
def test_pc(tts_and_tto):
    tts, tto, op_n_list = tts_and_tto
    tts = tts.copy()
    tts.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress_tdrk4)
    tts.compress_config = CompressConfig(CompressCriteria.fixed)
    check_result(tts, tto, 0.2, 5, op_n_list, 5e-4)