import torch

import quant_cuda

DEV = torch.device('cuda:0')

M = 4096
N = 4096

DTYPE = torch.float
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
vec = torch.randn((2, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((2, N), device=DEV, dtype=DTYPE)

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

mat = torch.randint(-1000000000, 1000000000, (M // 32 * 4, N), device=DEV, dtype=torch.int)
scales = torch.randn(N, device=DEV, dtype=DTYPE)
# zeros = torch.randn(N, device=DEV, dtype=DTYPE)
zeros = 7*torch.ones_like(scales)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    # vec = vec.reshape(1, 1, 4096)
    # mul = mul.reshape(-1)
    # scales,  zeros = scales.reshape(4096, 1), zeros.reshape(4096, 1)
    vec = vec.half()
    # print('0', vec, mat, mul, scales, zeros)
    quant_cuda.vecquant4matmul_faster(vec, mat, mul, scales, zeros)
    # print('1', mul)
    torch.cuda.synchronize()
print('mul', mul)
print('4it:', (time.time() - tick) / COUNT, '(faster)')

print('Verifiying kernel correctness ...')