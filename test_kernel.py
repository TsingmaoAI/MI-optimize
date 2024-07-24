import torch
import torch.nn as nn

import quant_cuda

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking OPT-175B FC2 matvec ...')

DEV = torch.device('cuda:0')

# M = 12288 * 4
# N = 12288
M = 4096
N = 4096

DTYPE = torch.half
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
vec = torch.randn((1, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((1, N), device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec, mat, out=mul) 
    torch.cuda.synchronize()
print('FP16:', (time.time() - tick) / COUNT)

DTYPE = torch.float
mat = mat.to(DTYPE)
vec = vec.to(DTYPE)
mul = mul.to(DTYPE)

# mat = torch.randint(-1000000000, 1000000000, (M // 1024 * 96, N), device=DEV, dtype=torch.int)
mat = torch.randint(-1000000000, 1000000000, (M // 32 * 4, N), device=DEV, dtype=torch.int)
scales = torch.randn(N, device=DEV, dtype=DTYPE)
zeros = torch.randn(N, device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros)
    torch.cuda.synchronize()
print('3bit:', (time.time() - tick) / COUNT)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    vec = vec.reshape(1, 1, 4096)
    mul = mul.reshape(-1)
    scales,  zeros = scales.reshape(4096, 1), zeros.reshape(4096, 1)
    quant_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros)
    torch.cuda.synchronize()
print('3bit:', (time.time() - tick) / COUNT, '(faster)')

print('Verifiying kernel correctness ...')