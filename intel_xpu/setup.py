"""
Setup script for Intel XPU Deep EP extension
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        # Build arguments
        build_args = ['--config', 'Release']
        
        # Parallel build
        if hasattr(self, 'parallel') and self.parallel:
            build_args += ['-j', str(self.parallel)]
        else:
            build_args += ['-j4']
        
        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        # Run CMake
        print(f"Running CMake in {build_temp}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        
        print(f"Building extension")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)


# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='deep_ep_xpu',
    version='0.1.0',
    author='Intel XPU Team',
    author_email='',
    description='Intel XPU Low Latency MoE Communication Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    ext_modules=[CMakeExtension('deep_ep_xpu._C')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[
        'torch>=2.0.0',
        'intel-extension-for-pytorch>=2.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
    zip_safe=False,
)

