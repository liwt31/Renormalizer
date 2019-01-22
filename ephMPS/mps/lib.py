# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
construct the operator matrix in the MPS sweep procedure
'''

import numpy as np
from ephMPS.lib import tensor as tensorlib


def GetLR(domain, siteidx, MPS, MPSconj, MPO, itensor=np.ones((1, 1, 1)), method="Scratch"):
    '''
    get the L/R Hamiltonian matrix at a random site(siteidx): 3d tensor
    S-     -S     MPSconj
    O- or  -O     MPO
    S-     -S     MPS
    enviroment part from disc,  system part from one step calculation
    support from scratch calculation: from two open boundary np.ones((1,1,1))
    '''

    assert domain in ["L", "R"]
    assert method in ["Enviro", "System", "Scratch"]

    if siteidx not in range(len(MPS)):
        return np.ones((1, 1, 1))

    if method == "Scratch":
        itensor = np.ones((1, 1, 1))
        if domain == "L":
            sitelist = range(siteidx + 1)
        else:
            sitelist = range(len(MPS) - 1, siteidx - 1, -1)
        for imps in sitelist:
            itensor = addone(itensor, MPS, MPSconj, MPO, imps, domain)
    elif method == "Enviro":
        itensor = Enviro_read(domain, siteidx)
    elif method == "System":
        itensor = addone(itensor, MPS, MPSconj, MPO, siteidx, domain)
        Enviro_write(domain, siteidx, itensor)

    return itensor


def addone(intensor, MPS, MPSconj, MPO, isite, domain):
    '''
    add one MPO/MPS(MPO) site
             _   _
            | | | |
    S-S-    | S-|-S-
    O-O- or | O-|-O- (the ancillary bond is traced)
    S-S-    | S-|-S-
            |_| |_|
    '''
    assert domain in ["L", "R", "l", "r"]

    if domain == "L" or domain == "l":
        assert intensor.shape[0] == MPSconj[isite].shape[0]
        assert intensor.shape[1] == MPO[isite].shape[0]
        assert intensor.shape[2] == MPS[isite].shape[0]
        '''
                       l 
        S-a-S-f    O-a-O-f
            d          d
        O-b-O-g or O-b-O-g  
            e          e
        S-c-S-h    O-c-O-h
                       l  
        '''

        if MPS[isite].ndim == 3:
            path = [([0, 1], "abc, adf -> bcdf"),
                    ([2, 0], "bcdf, bdeg -> cfeg"),
                    ([1, 0], "cfeg, ceh -> fgh")]
        elif MPS[isite].ndim == 4:
            path = [([0, 1], "abc, adlf -> bcdlf"),
                    ([2, 0], "bcdlf, bdeg -> clfeg"),
                    ([1, 0], "clfeg, celh -> fgh")]
        else:
            raise ValueError('MPS ndim at %d is not 3 or 4, got %s' % (isite, MPS[isite].ndim))
        outtensor = tensorlib.multi_tensor_contract(path, intensor, MPSconj[isite],
                                                    MPO[isite], MPS[isite])

    else:
        assert intensor.shape[0] == MPSconj[isite].shape[-1]
        assert intensor.shape[1] == MPO[isite].shape[-1]
        assert intensor.shape[2] == MPS[isite].shape[-1]
        '''
                       l
        -f-S-a-S    -f-S-a-S
           d           d
        -g-O-b-O or -g-O-b-O
           e           e
        -h-S-c-S    -h-S-c-S
                       l
        '''

        if MPS[isite].ndim == 3:
            path = [([0, 1], "fda, abc -> fdbc"),
                    ([2, 0], "fdbc, gdeb -> fcge"),
                    ([1, 0], "fcge, hec -> fgh")]
        elif MPS[isite].ndim == 4:
            path = [([0, 1], "fdla, abc -> fdlbc"),
                    ([2, 0], "fdlbc, gdeb -> flcge"),
                    ([1, 0], "flcge, helc -> fgh")]
        else:
            raise ValueError('MPS ndim at %d is not 3 or 4, got %s' % (isite, MPS[isite].ndim))
        outtensor = tensorlib.multi_tensor_contract(path, MPSconj[isite], intensor,
                                                    MPO[isite], MPS[isite])

    return outtensor


def construct_enviro(MPS, MPSconj, MPO, domain):
    tensor = np.ones((1, 1, 1))
    assert domain in ["L", "R", "l", "r"]
    if domain == "L" or domain == "l":
        start, end, inc = 0, len(MPS) - 1, 1
    else:
        start, end, inc = len(MPS) - 1, 0, -1

    for idx in range(start, end, inc):
        tensor = addone(tensor, MPS, MPSconj, MPO, idx, domain)
        Enviro_write(domain, idx, tensor)

# todo: real disk
virtual_disk = {}


def Enviro_write(domain, siteidx, tensor):
    #with open(domain + str(siteidx) + ".npy", 'wb') as f:
    #    np.save(f, tensor)
    virtual_disk[(domain, siteidx)] = tensor


def Enviro_read(domain, siteidx):
    #with open(domain + str(siteidx) + ".npy", 'rb') as f:
    #    return np.load(f)
    return virtual_disk[(domain, siteidx)]


def select_basis(qnset, Sset, qnlist, Mmax, percent=0):
    '''
    select basis according to Sset under qnlist requirement
    '''

    # convert to dict
    basdic = dict()
    for i in range(len(qnset)):
        # clean quantum number outside qnlist
        if qnset[i] in qnlist:
            basdic[i] = [qnset[i], Sset[i]]

    # each good quantum number block equally get percent/nblocks
    def block_select(basdic, qn, n):
        block_basdic = {i: basdic[i] for i in basdic if basdic[i][0] == qn}
        sort_block_basdic = sorted(block_basdic.items(), key=lambda x: x[1][1], reverse=True)
        nget = min(n, len(sort_block_basdic))
        # print(qn, "block # of retained basis", nget)
        sidx = [i[0] for i in sort_block_basdic[0:nget]]
        for idx in sidx:
            del basdic[idx]

        return sidx

    nbasis = min(len(basdic), Mmax)
    # print("# of selected basis", nbasis)
    sidx = []

    # equally select from each quantum number block
    if percent != 0:
        nbas_block = int(nbasis * percent / len(qnlist))
        for iqn in qnlist:
            sidx += block_select(basdic, iqn, nbas_block)

    # others
    nbasis = nbasis - len(sidx)

    sortbasdic = sorted(basdic.items(), key=lambda x: x[1][1], reverse=True)
    sidx += [i[0] for i in sortbasdic[0:nbasis]]

    assert len(sidx) == len(set(sidx))  # there must be no duplicated

    return sidx


def updatemps(vset, sset, qnset, compset, nexciton, Mmax, percent=0):
    '''
    select basis to construct new mps, and complementary mps
    vset, compset is the column vector
    '''
    sidx = select_basis(qnset, sset, range(nexciton + 1), Mmax, percent=percent)
    mpsdim = len(sidx)
    mps = np.zeros((vset.shape[0], mpsdim), dtype=vset.dtype)

    if compset is not None:
        compmps = np.zeros((compset.shape[0], mpsdim), dtype=compset.dtype)
    else:
        compmps = None

    mpsqn = []
    stot = 0.0
    for idim in range(mpsdim):
        mps[:, idim] = vset[:, sidx[idim]].copy()
        if (compset is not None) and sidx[idim] < compset.shape[1]:
            compmps[:, idim] = compset[:, sidx[idim]].copy() * sset[sidx[idim]]
        mpsqn.append(qnset[sidx[idim]])
        stot += sset[sidx[idim]] ** 2

    # print("discard:", 1.0 - stot)

    return mps, mpsdim, mpsqn, compmps