import os
from pathlib import Path
from setuptools import setup, Extension, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT_DIR = os.path.dirname(__file__)

def _is_cuda() -> bool:
      return (torch.version.cuda is not None)

ext_modules = []

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = ["-O2", "-std=c++17"]

def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

if _is_cuda():
    ext_modules.append(
        CUDAExtension(
            name="torch_tools._C",
            sources=glob("src/extensions/*.cu") + glob("src/extensions/*.cpp"),
            extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
            },
        )
    )

    # it seems cannot set level name more than 3? e.g.:
    # ext_modules.append(
    #     CUDAExtension(
    #         name="torch_tools._C.add",
    #         sources=glob("src/add/*.cu") + glob("src/add/*.cpp"),
    #         extra_compile_args={
    #         "cxx": CXX_FLAGS,
    #         "nvcc": NVCC_FLAGS,
    #         },
    #     )
    # )

packages_list = [
    "torch_tools",
    "torch_tools._C",
    "torch_tools.ops",
]
      
setup(
    name='my-torch-tools',
    version="1.0.0",
    packages=packages_list,
    package_dir={
        '':'src',
        },
    package_data={
        # '_C':['*.pyi']
    },
    setup_requires=['torch'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)