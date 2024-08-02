import time
import torch
import quant_cuda  


def test_int2GroupWeightExtraction():
    M = 1024
    N = 1024
    group = 1024

    A = torch.randint(-1000000000, 1000000000, (M , N // 32 * 2), device='cuda', dtype=torch.int)
    scales = torch.randn(M, N//group, dtype=torch.float32, device='cuda')
    zeros = torch.randn(M, N//group, dtype=torch.float32, device='cuda')
    B = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    expected_B = torch.zeros_like(B)
    for i in range(M):
        for j in range(N//16):
            scale = scales[i, (j*16 // group)]
            zero = zeros[i, (j*16 // group)]
            tmp = int(A[i, j])
            for k in range(16):
                expected_B[i, j*16 + k] = ((((tmp >> (2*(15-k))) & 0x3) - zero) * scale)

    start_time = time.time()
    for _ in range(1000):  
        quant_cuda.int2GroupWeightExtraction(A, scales, zeros, B, group)
        torch.cuda.synchronize()  # Synchronize the GPU with the CPU
    end_time = time.time()
    print(f"Average execution time: {(end_time - start_time) / 1000:.6f} seconds")

    quant_cuda.int2GroupWeightExtraction(A, scales, zeros, B, group)
    
    if torch.allclose(B, expected_B, atol=1e-6):
        print("Accuracy test passed")
    else:
        print("Accuracy test failed")
        print("Expected:", expected_B)
        print("Got:", B)

if __name__ == "__main__":
    test_int2GroupWeightExtraction()