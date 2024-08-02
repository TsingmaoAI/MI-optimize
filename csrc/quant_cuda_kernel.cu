#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

const int BLOCKWIDTH  = 256;
// const int BLOCKHEIGHT =  24;
const int BLOCKHEIGHT  = 16;
const int BLOCKWIDTH_INT4 = 16;
const int BLOCKWIDTH_INT2 = 32;

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
);


__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
);

template <typename scalar_t>
__global__ void int4GroupWeightExtractKernel(
    const  int* __restrict__ A,       
    const  scalar_t* __restrict__ scales,  
    const  scalar_t* __restrict__ zeros,
           scalar_t* __restrict__ B, 
    int height,
    int width,
    int group,
    int veclen
);

template <typename scalar_t>
__global__ void int2GroupWeightExtractKernel(
    const  int* __restrict__ A,       
    const  scalar_t* __restrict__ scales,  
    const  scalar_t* __restrict__ zeros,
           scalar_t* __restrict__ B, 
    int height,
    int width,
    int group,
    int veclen
);

void vecquant4matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<float>(),
    height, width
  );
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[16][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 16; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  half2 scale = __float2half2_rn(scales[col]);
  half2 zero = __float2half2_rn(-zeros[col]);

  int i = width * row + col;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;

  __syncthreads();

  while (k < blockwidth2) {
    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 28) & 0xF][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0xF][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 20) & 0xF][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 16) & 0xF][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0xF][off], scale, zero), blockvec[k + 4], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  8) & 0xF][off], scale, zero), blockvec[k + 5], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  4) & 0xF][off], scale, zero), blockvec[k + 6], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0xF][off], scale, zero), blockvec[k + 7], res2);
    i += width;
    k += 8;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[col], res);
}

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        height, width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  if (row >= height || col >= width) {
    return;  // Prevent out-of-bounds access
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  if (threadIdx.x < BLOCKWIDTH && (row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x < width) {
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  } else {
    blockvec[threadIdx.x] = 0.0;
  }
  __syncthreads();

  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];

  scalar_t res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp1;

  while (k < BLOCKWIDTH) {
    if (i >= height * width) break;  // Prevent out-of-bounds access
    tmp1 = as_unsigned(mat[i]);
    res += (scalar_t((tmp1 >> 28) & 0xF) - zero) * scale * blockvec[k + 0];
    res += (scalar_t((tmp1 >> 24) & 0xF) - zero) * scale * blockvec[k + 1];
    res += (scalar_t((tmp1 >> 20) & 0xF) - zero) * scale * blockvec[k + 2];
    res += (scalar_t((tmp1 >> 16) & 0xF) - zero) * scale * blockvec[k + 3];
    res += (scalar_t((tmp1 >> 12) & 0xF) - zero) * scale * blockvec[k + 4];
    res += (scalar_t((tmp1 >>  8) & 0xF) - zero) * scale * blockvec[k + 5];
    res += (scalar_t((tmp1 >>  4) & 0xF) - zero) * scale * blockvec[k + 6];
    res += (scalar_t((tmp1 >>  0) & 0xF) - zero) * scale * blockvec[k + 7];
    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}

void int4GroupWeightExtraction_cuda(
    torch::Tensor A,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor B,
    int group
) {
    int size_m = A.size(0);  // 4096 
    int size_n = A.size(1);  // 4096 / 32 * 4
    int veclen = B.size(1) / group;

    dim3 blocks(
        (size_m + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (size_n + BLOCKWIDTH_INT4 - 1) / BLOCKWIDTH_INT4
    );
    dim3 threads(BLOCKHEIGHT, BLOCKWIDTH_INT4);

    AT_DISPATCH_FLOATING_TYPES(
        B.type(), "int4gWeightExtraction_cuda", ([&] {
            int4GroupWeightExtractKernel<<<blocks, threads>>>(
                A.data<int>(), scales.data<scalar_t>(),
                zeros.data<scalar_t>(), B.data<scalar_t>(), 
                size_m, size_n, group, veclen
            );
        })
    );
}

template <typename scalar_t>
__global__ void int4GroupWeightExtractKernel(
    const  int* __restrict__ A,       
    const  scalar_t* __restrict__ scales,  
    const  scalar_t* __restrict__ zeros,
           scalar_t* __restrict__ B, 
    int height,
    int width,
    int group,
    int veclen
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int t = row * veclen  + (col * 8 / group);
        scalar_t scale = scales[t];
        scalar_t zero = zeros[t];
        int i = row * width + col;

        unsigned int tmp = A[i];
        B[i * 8 + 0] = ((scalar_t((tmp >> 28) & 0xF) - zero) * scale);
        B[i * 8 + 1] = ((scalar_t((tmp >> 24) & 0xF) - zero) * scale);
        B[i * 8 + 2] = ((scalar_t((tmp >> 20) & 0xF) - zero) * scale);
        B[i * 8 + 3] = ((scalar_t((tmp >> 16) & 0xF) - zero) * scale);
        B[i * 8 + 4] = ((scalar_t((tmp >> 12) & 0xF) - zero) * scale);
        B[i * 8 + 5] = ((scalar_t((tmp >>  8) & 0xF) - zero) * scale);
        B[i * 8 + 6] = ((scalar_t((tmp >>  4) & 0xF) - zero) * scale);
        B[i * 8 + 7] = ((scalar_t((tmp >>  0) & 0xF) - zero) * scale);
    }
}

void int2GroupWeightExtraction_cuda(
    torch::Tensor A,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor B,
    int group
) {
    int size_m = A.size(0);  // 4096 
    int size_n = A.size(1);  // 4096 / 32 * 2
    int veclen = B.size(1) / group;

    dim3 blocks(
        (size_m + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
        (size_n + BLOCKWIDTH_INT2 - 1) / BLOCKWIDTH_INT2
    );
    dim3 threads(BLOCKHEIGHT, BLOCKWIDTH_INT2);

    AT_DISPATCH_FLOATING_TYPES(
        B.type(), "int2GroupWeightExtraction_cuda", ([&] {
            int2GroupWeightExtractKernel<<<blocks, threads>>>(
                A.data<int>(), scales.data<scalar_t>(),
                zeros.data<scalar_t>(), B.data<scalar_t>(), 
                size_m, size_n, group, veclen
            );
        })
    );
}

template <typename scalar_t>
__global__ void int2GroupWeightExtractKernel(
    const  int* __restrict__ A,       
    const  scalar_t* __restrict__ scales,  
    const  scalar_t* __restrict__ zeros,
           scalar_t* __restrict__ B, 
    int height,
    int width,
    int group,
    int veclen
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int t = row * veclen  + (col * 16 / group);
        scalar_t scale = scales[t];
        scalar_t zero = zeros[t];
        int i = row * width + col;

        unsigned int tmp = A[i];
        B[i * 16 + 0] = ((scalar_t((tmp >> 30) & 0x3) - zero) * scale);
        B[i * 16 + 1] = ((scalar_t((tmp >> 28) & 0x3) - zero) * scale);
        B[i * 16 + 2] = ((scalar_t((tmp >> 26) & 0x3) - zero) * scale);
        B[i * 16 + 3] = ((scalar_t((tmp >> 24) & 0x3) - zero) * scale);
        B[i * 16 + 4] = ((scalar_t((tmp >> 22) & 0x3) - zero) * scale);
        B[i * 16 + 5] = ((scalar_t((tmp >> 20) & 0x3) - zero) * scale);
        B[i * 16 + 6] = ((scalar_t((tmp >> 18) & 0x3) - zero) * scale);
        B[i * 16 + 7] = ((scalar_t((tmp >> 16) & 0x3) - zero) * scale);
        B[i * 16 + 8] = ((scalar_t((tmp >> 14) & 0x3) - zero) * scale);
        B[i * 16 + 9] = ((scalar_t((tmp >> 12) & 0x3) - zero) * scale);
        B[i * 16 + 10] = ((scalar_t((tmp >> 10) & 0x3) - zero) * scale);
        B[i * 16 + 11] = ((scalar_t((tmp >> 8) & 0x3) - zero) * scale);
        B[i * 16 + 12] = ((scalar_t((tmp >> 6) & 0x3) - zero) * scale);
        B[i * 16 + 13] = ((scalar_t((tmp >> 4) & 0x3) - zero) * scale);
        B[i * 16 + 14] = ((scalar_t((tmp >> 2) & 0x3) - zero) * scale);
        B[i * 16 + 15] = ((scalar_t((tmp >> 0) & 0x3) - zero) * scale);
    }
}