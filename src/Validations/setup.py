# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hdc_cuda',
    ext_modules=[
        CUDAExtension(
            'hdc_cuda',
            ['hamming_wrapper.cpp', 'hamming_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],  # aggressive optimization
                'nvcc': [
                    '-O3',
                    '--use_fast_math',             # enable fast math (can slightly affect numerical precision)
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
