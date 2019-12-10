from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='marching_cubes_cpp',
    ext_modules=[
        CppExtension('marching_cubes_cpp', ['marching_cubes.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
