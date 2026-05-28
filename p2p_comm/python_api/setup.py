"""
Setup script for heteromem_p2p Python package
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

def get_pybind_include():
    """Helper to get pybind11 include path"""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        # Return empty string if not installed yet
        # Will be handled by setup_requires
        return ""

# Paths
XRT_PATH = os.environ.get('XILINX_XRT', '/opt/xilinx/xrt')
ROCM_PATH = os.environ.get('ROCM_PATH', '/opt/rocm')

ext_modules = [
    Extension(
        'heteromem_p2p',
        sources=[
            'src/bindings.cpp',
            'src/p2p_manager.cpp',
        ],
        include_dirs=[
            'src',
            f'{XRT_PATH}/include',
            f'{ROCM_PATH}/include',
        ],
        library_dirs=[
            f'{XRT_PATH}/lib',
            f'{ROCM_PATH}/lib',
        ],
        libraries=[
            'xrt_coreutil',
            'amdhip64',
            'uuid',
            'pthread',
        ],
        language='c++',
        extra_compile_args=[
            '-std=c++17',
            '-O3',
            '-Wall',
            '-fPIC',
            '-D__HIP_PLATFORM_AMD__',
        ],
    ),
]

class BuildExt(build_ext):
    """Custom build extension to add pybind11 include path"""
    def build_extensions(self):
        # Add pybind11 include path
        pybind_include = get_pybind_include()
        if pybind_include:
            for ext in self.extensions:
                ext.include_dirs.insert(0, pybind_include)
        build_ext.build_extensions(self)

setup(
    name='heteromem_p2p',
    version='0.1.0',
    author='HeteroMem Team',
    author_email='your.email@example.com',
    description='FPGA-GPU P2P Communication Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    setup_requires=[
        'pybind11>=2.6.0',
    ],
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'scipy>=1.5.0',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False,
)
