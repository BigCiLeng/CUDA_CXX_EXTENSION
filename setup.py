'''
Author: BigCiLeng && bigcileng@outlook.com
Date: 2023-10-27 20:18:39
LastEditors: BigCiLeng && bigcileng@outlook.com
LastEditTime: 2023-10-28 23:06:47
FilePath: /cuda_cxx_extension/setup.py
Description: 

Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
'''
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='cppcuda_tutorial',
    version='1.0',
    author='bigcileng',
    author_email='bigcileng@outlook.com',
    description='cppcuda example',
    long_description='cppcude example',
    ext_modules=[
        CUDAExtension(
            name='cppcuda_tutorial',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                               'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
