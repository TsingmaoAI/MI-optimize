#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>


void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
); 

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

void int4GroupWeightExtraction_cuda(
    torch::Tensor inputs, torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor outputs, int group
);

void int2GroupWeightExtraction_cuda(
    torch::Tensor inputs, torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor outputs, int group
);

void int4GroupWeightExtraction(
    torch::Tensor inputs, torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor outputs, int group
){
    if (inputs.type().is_cuda()){
        int4GroupWeightExtraction_cuda(inputs, scales, zeros, outputs, group);
    }else{
        throw std::runtime_error("int4gWeightExtraction is only supported on GPU.");
    }
}

void int2GroupWeightExtraction(
    torch::Tensor inputs, torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor outputs, int group
){
    if (inputs.type().is_cuda()){
        int2GroupWeightExtraction_cuda(inputs, scales, zeros, outputs, group);
    }else{
        throw std::runtime_error("int4gWeightExtraction is only supported on GPU.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul_faster", &vecquant4matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("int4GroupWeightExtraction", &int4GroupWeightExtraction, "group-wise quantized int4 weight extraction");
  m.def("int2GroupWeightExtraction", &int2GroupWeightExtraction, "group-wise quantized int4 weight extraction");
}