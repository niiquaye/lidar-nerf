# import os
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# _src_path = os.path.dirname(os.path.abspath(__file__))
# import torch 
# abi_flag = "0" #"1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
# # os.environ["CXXFLAGS"] = f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"
# os.environ["CC"] = "/usr/bin/gcc-11"
# os.environ["CXX"] = "/usr/bin/g++-11"
# nvcc_flags = [
#     "-O2",
#     "-std=c++17",
#     "-U__CUDA_NO_HALF_OPERATORS__",
#     "-U__CUDA_NO_HALF_CONVERSIONS__",
#     "-U__CUDA_NO_HALF2_OPERATORS__",
#     f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
# ]

# if os.name == "posix":
#     c_flags = ["-O2", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"]
# elif os.name == "nt":
#     c_flags = ["/O2", "/std:c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"]

#     # find cl.exe
#     def find_cl_path():
#         import glob

#         for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
#             paths = sorted(
#                 glob.glob(
#                     r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"
#                     % edition
#                 ),
#                 reverse=True,
#             )
#             if paths:
#                 return paths[0]

#     # If cl.exe is not on path, try to find it.
#     if os.system("where cl.exe >nul 2>nul") != 0:
#         cl_path = find_cl_path()
#         if cl_path is None:
#             raise RuntimeError(
#                 "Could not locate a supported Microsoft Visual C++ installation"
#             )
#         os.environ["PATH"] += ";" + cl_path

# setup(
#     name="gridencoder",  # package name, import this to use python API
#     ext_modules=[
#         CUDAExtension(
#             name="_gridencoder",  # extension name, import this to use CUDA API
#             sources=[
#                 os.path.join(_src_path, "src", f)
#                 for f in [
#                     "gridencoder.cu",
#                     "bindings.cpp",
#                 ]
#             ],
#             extra_compile_args={
#                 "cxx": c_flags,
#                 "nvcc": nvcc_flags,
#             },
#         ),
#     ],
#     cmdclass={
#         "build_ext": BuildExtension,
#     },
# )


# import os
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# abi_flag = "0"  # Must match `torch._C._GLIBCXX_USE_CXX11_ABI`

# setup(
#     name="gridencoder",
#     ext_modules=[
#         CUDAExtension(
#             name="_gridencoder",
#             sources=[
#                 "src/gridencoder.cu",
#                 "src/bindings.cpp",
#             ],
#             extra_compile_args={
#                 "cxx": [
#                     "-O2",
#                     "-std=c++17",
#                     f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
#                 ],
#                 "nvcc": [
#                     "-O2",
#                     "-std=c++17",
#                     # "--expt-relaxed-constexpr",
#                     f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
#                     # "-gencode=arch=compute_75,code=sm_75",
#                     "-U__CUDA_NO_HALF_OPERATORS__",
#                     "-U__CUDA_NO_HALF_CONVERSIONS__",
#                     "-U__CUDA_NO_HALF2_OPERATORS__",
#                 ]
#             },
#         ),
#     ],
#     cmdclass={"build_ext": BuildExtension},
# )

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths

# Match PyTorch ABI
abi_flag = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
libtorch_path = library_paths()[0]

setup(
    name="gridencoder",
    ext_modules=[
        CUDAExtension(
            name="_gridencoder",
            sources=[
                "src/gridencoder.cu",
                "src/bindings.cpp",
            ],
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-std=c++17",
                    f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
                ],
                "nvcc": [
                    "-O2",
                    "-std=c++17",
                    f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode=arch=compute_75,code=sm_75",
                ]
            },
            extra_link_args=[
                f"-Wl,-rpath,{libtorch_path}"  # 👈 This is key
            ]
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)