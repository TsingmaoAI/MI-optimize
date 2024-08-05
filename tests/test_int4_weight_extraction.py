import time
import torch
import quant_cuda  


def test_int4GroupWeightExtraction():
    M = 1024
    N = 1024
    group = 1024

    A = torch.randint(-1000000000, 1000000000, (M , N // 32 *4), device='cuda', dtype=torch.int)
    scales = torch.randn(M, N//group, dtype=torch.float32, device='cuda')
    zeros = torch.randn(M, N//group, dtype=torch.float32, device='cuda')
    B = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    expected_B = torch.zeros_like(B)
    for i in range(M):
        for j in range(N//8):
            scale = scales[i, (j*8 // group)]
            zero = zeros[i, (j*8 // group)]
            tmp = int(A[i, j])
            for k in range(8):
                expected_B[i , j*8 + k] = ((((tmp >> (4*(7-k))) & 0xF) - zero) * scale)
        
    start_time = time.time()
    for _ in range(1000):  
        quant_cuda.int4GroupWeightExtraction(A, scales, zeros, B, group)
        torch.cuda.synchronize()  # Synchronize the GPU with the CPU
    end_time = time.time()
    print(f"Average execution time: {(end_time - start_time) / 1000:.6f} seconds")

    quant_cuda.int4GroupWeightExtraction(A, scales, zeros, B, group)
    
    if torch.allclose(B, expected_B, atol=1e-6):
        print("Accuracy test passed")
    else:
        print("Accuracy test failed")
        print("Expected:", expected_B)
        print("Got:", B)

if __name__ == "__main__":
    test_int4GroupWeightExtraction()
