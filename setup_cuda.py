from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['csrc/quant_cuda.cpp', 'csrc/quant_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)