import importlib.util
import logging
import os
import subprocess
import sys
from sysconfig import get_paths
from typing import Dict, List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def check_or_set_default_env(cmake_args,
                             env_name,
                             env_variable,
                             default_path=""):
    if env_variable is None:
        logging.warning(
            f"No {env_name} found in your environment, pleause try to set {env_name} "
            "if you customize the installation path of this library, otherwise default "
            "path will be adapted during build this project")
        logging.warning(f"Set default {env_name}: {default_path}")
        env_variable = default_path
    else:
        logging.info(f"Found existing {env_name}: {env_variable}")
    # cann package seems will check this environments in cmake, need write this env variable back.
    cmake_args += [f"-D{env_name}={env_variable}"]
    return cmake_args

envs = load_module_from_path("envs",
                             os.path.join(ROOT_DIR, "vllm_tcu", "envs.py"))

class CMakeExtension(Extension):

    def __init__(self,
                 name: str,
                 cmake_lists_dir: str = ".",
                 **kwargs) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()
        num_jobs = max(1, num_jobs)

        return num_jobs

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        source_dir = os.path.abspath(ROOT_DIR)
        python_executable = sys.executable
        cmake_args = ["cmake"]
        # Default use release mode to compile the csrc code
        # Turbo now support compiled with Release, Debug and RelWithDebugInfo
        if envs.CMAKE_BUILD_TYPE is None or envs.CMAKE_BUILD_TYPE not in [
            "Debug",
            "Release",
            "RelWithDebugInfo",
        ]:
            envs.CMAKE_BUILD_TYPE = "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={envs.CMAKE_BUILD_TYPE}"]
        # Default dump the compile commands for lsp
        cmake_args += ["-DCMAKE_EXPORT_COMPILE_COMMANDS=1"]
        if envs.CXX_COMPILER is not None:
            cmake_args += [f"-DCMAKE_CXX_COMPILER={envs.CXX_COMPILER}"]
        if envs.C_COMPILER is not None:
            cmake_args += [f"-DCMAKE_C_COMPILER={envs.C_COMPILER}"]
        if envs.VERBOSE:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        # find PYTHON_EXECUTABLE
        check_or_set_default_env(cmake_args, "PYTHON_EXECUTABLE",
                                 sys.executable)

        # find PYTHON_INCLUDE_PATH
        check_or_set_default_env(cmake_args, "PYTHON_INCLUDE_PATH",
                                 get_paths()["include"])

        install_path = os.path.join(ROOT_DIR, self.build_lib)
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = os.path.join(ROOT_DIR, "vllm_tcu")
        # add CMAKE_INSTALL_PATH
        cmake_args += [f"-DCMAKE_INSTALL_PREFIX={install_path}"]
        build_tool = []

        cmake_args += [source_dir]
        logging.info(f"cmake config command: {cmake_args}")
        try:
            subprocess.check_call(cmake_args, cwd=self.build_temp)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake configuration failed: {e}")

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        if not envs.COMPILE_CUSTOM_KERNELS:
            return
        # 确保 CMake 可用
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError(f"Cannot find CMake executable: {e}")

        # 创建构建目录
        os.makedirs(self.build_temp, exist_ok=True)

        # 配置项目（使用统一的安装路径）
        config_cmd = [
            "cmake",
            "-S", ".",
            "-B", self.build_temp,
            f"-DCMAKE_INSTALL_PREFIX={os.path.abspath(os.path.join(ROOT_DIR, self.build_lib))}"
        ]
        subprocess.check_call(config_cmd, cwd=ROOT_DIR)

        # 构建项目
        build_cmd = ["cmake", "--build", self.build_temp]
        subprocess.check_call(build_cmd)

        # 安装到 setuptools 期望的位置
        install_cmd = ["cmake", "--install", self.build_temp]
        subprocess.check_call(install_cmd)

        print("Installation completed successfully")


    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


class custom_install(install):

    def run(self):
        self.run_command("build_ext")
        install.run(self)


ext_modules = []
ext_modules = [CMakeExtension(name="vllm_tcu._C")]

cmdclass = {"build_ext": cmake_build_ext, "install": custom_install}

setup(
    name="vllm_tcu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm>=0.8.0"
    ],
    entry_points={
        "vllm.general_plugins": [
            "register_custom_models = vllm_tcu.models:register_custom_models"
        ],
        "vllm.platform_plugins": [
            "register_platform_plugins = vllm_tcu:register_platform_plugins"
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
