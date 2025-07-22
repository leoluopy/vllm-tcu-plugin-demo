from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
debug_mode = True
debug_compile_args = ['-g', '-O0'] if debug_mode else []
debug_link_args = ['-g'] if debug_mode else []

setup(name='torch_tcu',
      ext_modules=[cpp_extension.CppExtension('torch_tcu',
                                              ['torch_tcu.cpp',
                                               'op_impl_cpu/op_implements.cpp'
                                               ],
                                              include_dirs=[
                                                  current_dir,  # 相当于 "./"
                                                  os.path.join(current_dir, 'op_impl_cpu')  # 相当于 "op_impl_cpu"
                                              ],
                                              extra_compile_args=debug_compile_args,  # 添加调试编译选项
                                              extra_link_args=debug_link_args  # 添加调试链接选项
                                              )],

      cmdclass={'build_ext': cpp_extension.BuildExtension})
