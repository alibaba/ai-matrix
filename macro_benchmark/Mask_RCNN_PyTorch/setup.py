# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os
import copy
import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "vision.cpp"))
    main_file_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    source_cuda_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cu"))
    source_cpp_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cpp"))

    sources = main_file + source_cpu
    sources_nhwc = main_file_nhwc + source_cuda_nhwc + source_cpp_nhwc
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        extension("maskrcnn_benchmark.NHWC",
            sources_nhwc,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=copy.deepcopy(extra_compile_args),
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
