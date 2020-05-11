##
import os
import copy
import ctypes
from ctypes import *
import cupy as cp

import numpy as np
import matplotlib.pyplot as plt

##
pnsz = np.asarray([100, 50, 30], dtype=np.int32)

mul = np.random.randn()
add = np.random.randn()

src = np.random.randn(pnsz[0], pnsz[1], pnsz[2]).astype(dtype=np.float32)

## Numpy in CPU
src_np = copy.deepcopy(src)
dst_np = src_np

dst_np = dst_np * mul
dst_np = dst_np + add
dst_np = dst_np * dst_np
dst_np = dst_np + dst_np

## Clang in CPU
clang_file = os.path.join(os.path.dirname(__file__), 'libmath_clang.so')
_math_clang = ctypes.CDLL(clang_file)

__mul_const_clang = _math_clang.mul_const
__add_const_clang = _math_clang.add_const
__mul_mat_clang = _math_clang.mul_mat
__add_mat_clang = _math_clang.add_mat

__mul_const_clang.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__mul_const_clang.restypes = c_void_p
__add_const_clang.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__add_const_clang.restypes = c_void_p
__mul_mat_clang.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__mul_mat_clang.restypes = c_void_p
__add_mat_clang.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__add_mat_clang.restypes = c_void_p

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))

src_clang = copy.deepcopy(src)
dst_clang = src_clang

__mul_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), mul, c_int_p(pnsz))
__add_const_clang(c_float_p(dst_clang), c_float_p(dst_clang), add, c_int_p(pnsz))
__mul_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))
__add_mat_clang(c_float_p(dst_clang), c_float_p(dst_clang), c_float_p(dst_clang), c_int_p(pnsz))

## CU in GPU
cu_file = os.path.join(os.path.dirname(__file__), 'libmath_cu.so')
_math_cu = ctypes.CDLL(cu_file)

__mul_const_cu = _math_cu.mul_const
__add_const_cu = _math_cu.add_const
__mul_mat_cu = _math_cu.mul_mat
__add_mat_cu = _math_cu.add_mat

__mul_const_cu.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__mul_const_cu.restypes = c_void_p
__add_const_cu.argtypes = (POINTER(c_float), POINTER(c_float), c_float, POINTER(c_int))
__add_const_cu.restypes = c_void_p
__mul_mat_cu.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__mul_mat_cu.restypes = c_void_p
__add_mat_cu.argtypes = (POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_int))
__add_mat_cu.restypes = c_void_p

c_float_p = lambda x: x.ctypes.data_as(POINTER(c_float))
c_int_p = lambda x: x.ctypes.data_as(POINTER(c_int))

src_cu = copy.deepcopy(src)
dst_cu = src_cu

__mul_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), mul, c_int_p(pnsz))
__add_const_cu(c_float_p(dst_cu), c_float_p(dst_cu), add, c_int_p(pnsz))
__mul_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))
__add_mat_cu(c_float_p(dst_cu), c_float_p(dst_cu), c_float_p(dst_cu), c_int_p(pnsz))

# print(dst_np[:10, 0, 0])
# print(dst_cu[:10, 0, 0])
#
# print(np.sum(dst_np - dst_cu))

## Cupy in GPU
src_cp = cp.asarray(copy.deepcopy(src))
dst_cp = src_cp

mul_cp = cp.float32(mul)
add_cp = cp.float32(add)

dst_cp = dst_cp * mul_cp
dst_cp = dst_cp + add_cp
dst_cp = dst_cp * dst_cp
dst_cp = dst_cp + dst_cp

dst_cp2np = cp.asnumpy(dst_cp)

# print(dst_np[:10, 0, 0])
# print(dst_cp[:10, 0, 0])
# print(np.sum(dst_np - dst_cp2np))

## Cupy with kernel in GPU
cu_file = os.path.join(os.path.dirname(__file__), 'math_cu.cuh')
with open(cu_file, 'r') as f:
    code = f.read()

__mul_const_cp = cp.RawKernel(code, 'mul_const_kernel')
__add_const_cp = cp.RawKernel(code, 'add_const_kernel')
__mul_mat_cp = cp.RawKernel(code, 'mul_mat_kernel')
__add_mat_cp = cp.RawKernel(code, 'add_mat_kernel')

nthread = 8
nblock = (nthread, nthread, nthread)
ngrid = (int((pnsz[0] + nthread - 1)/nthread),
         int((pnsz[1] + nthread - 1)/nthread),
         int((pnsz[2] + nthread - 1)/nthread))

src_cph = cp.asarray(copy.deepcopy(src))
dst_cph = src_cph
pnsz_cph = cp.asarray(copy.deepcopy(pnsz))

mul_cph = cp.float32(mul)
add_cph = cp.float32(add)

__mul_const_cp(ngrid, nblock, args=(dst_cph, dst_cph, mul_cph, pnsz_cph))
__add_const_cp(ngrid, nblock, args=(dst_cph, dst_cph, add_cph, pnsz_cph))
__mul_mat_cp(ngrid, nblock, args=(dst_cph, dst_cph, dst_cph, pnsz_cph))
__add_mat_cp(ngrid, nblock, args=(dst_cph, dst_cph, dst_cph, pnsz_cph))

dst_cph2np = cp.asnumpy(dst_cph)


##
print("Error between numpy and clang: %.8f" % np.sum(dst_np - dst_clang))
print("Error between numpy and cu   : %.8f" % np.sum(dst_np - dst_cu))
print("Error between numpy and cupy : %.8f" % np.sum(dst_np - dst_cp2np))
print("Error between numpy and cupyh: %.8f" % np.sum(dst_np - dst_cph2np))






























