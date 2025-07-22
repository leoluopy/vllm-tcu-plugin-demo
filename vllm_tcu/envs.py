
import os
from typing import Any, Callable, Dict

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: Dict[str, Callable[[], Any]] = {
    "MAX_JOBS":
        lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE":
        lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "COMPILE_CUSTOM_KERNELS":
        lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    "CXX_COMPILER":
        lambda: os.getenv("CXX_COMPILER", None),
    "C_COMPILER":
        lambda: os.getenv("C_COMPILER", None),
    "VLLM_VERSION":
        lambda: os.getenv("VLLM_VERSION", None),
    "VERBOSE":
        lambda: bool(int(os.getenv('VERBOSE', '0'))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())