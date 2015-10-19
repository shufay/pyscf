#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Analytical integration
# J. Chem. Phys. 65, 3826
# J. Chem. Phys. 111, 8778
# J. Comput. Phys. 44, 289
#
# Numerical integration
# J. Comput. Chem. 27, 1009
# Chem. Phys. Lett. 296, 445
#

import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor

libecp = moleintor._cint

def intor(mol):
    nao = mol.nao_nr()
    mat = numpy.zeros((nao,nao))
    ip = 0
    for ish in range(mol.nbas):
        jp = 0
        for jsh in range(ish+1):
            buf = type1_by_shell(mol, (ish,jsh))
            di, dj = buf.shape
            mat[ip:ip+di,jp:jp+dj] += buf

            buf = type2_by_shell(mol, (ish,jsh))
            di, dj = buf.shape
            mat[ip:ip+di,jp:jp+dj] += buf
            jp += dj
        ip += di
    return lib.hermi_triu(mat)

def type1_by_shell(mol, shls):
    li = mol.bas_angular(shls[0])
    lj = mol.bas_angular(shls[1])
    di = (li*2+1) * mol.bas_nctr(shls[0])
    dj = (lj*2+1) * mol.bas_nctr(shls[1])
    buf = numpy.empty((di,dj), order='F')
    libecp.ECPtype1_sph(buf.ctypes.data_as(ctypes.c_void_p),
                        (ctypes.c_int*2)(*shls),
                        mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(len(mol._ecpbas)),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                        mol._env.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_void_p())
    return buf

def type2_by_shell(mol, shls):
    li = mol.bas_angular(shls[0])
    lj = mol.bas_angular(shls[1])
    di = (li*2+1) * mol.bas_nctr(shls[0])
    dj = (lj*2+1) * mol.bas_nctr(shls[1])
    buf = numpy.empty((di,dj), order='F')
    libecp.ECPtype2_sph(buf.ctypes.data_as(ctypes.c_void_p),
                        (ctypes.c_int*2)(*shls),
                        mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(len(mol._ecpbas)),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                        mol._env.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_void_p())
    return buf

def core_configuration(nelec_core):
    conf_dic = {
        0 : '0s0p0d0f',
        2 : '1s0p0d0f',
        10: '2s1p0d0f',
        18: '3s2p0d0f',
        28: '3s2p1d0f',
        36: '4s3p1d0f',
        46: '4s3p2d0f',
        54: '5s4p2d0f',
        60: '4s3p2d1f',
        68: '5s4p2d1f',
        78: '5s4p3d1f',
        92: '5s4p3d2f',
    }
    coreshell = [int(x) for x in conf_dic[nelec_core][::2]]
    return coreshell


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M(atom='''
 Cu 0. 0. 0.
 H  0.  0. -1.56
 H  0.  0.  1.56
''',
                basis={'Cu':'lanl2dz', 'H':'sto3g'},
                ecp = {'cu':'lanl2dz'},
                #basis={'Cu':'crenbs', 'H':'sto3g'},
                #ecp = {'cu':'crenbs'},
                charge=-1,
                verbose=4)
    mf = scf.RHF(mol)
    print(mf.kernel(), -196.09477546034623)

    mol = gto.M(atom='''
 Na 0. 0. 0.
 H  0.  0.  1.
''',
                basis={'Na':'lanl2dz', 'H':'sto3g'},
                ecp = {'Na':'lanl2dz'},
                verbose=0)
    mf = scf.RHF(mol)
    print(mf.kernel(), -0.45002315562861461)

