cmake_minimum_required(VERSION 3.16)
project(vllm_tcu_C)



set(CMAKE_CXX_STANDARD 17)
# use python , pybind.__file__ find
list(APPEND CMAKE_PREFIX_PATH "/opt/conda/lib/python3.11/site-packages/pybind11/share/cmake/pybind11/")
list(APPEND CMAKE_PREFIX_PATH "/opt/conda/lib/python3.11/site-packages/torch/share/cmake/Torch/")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)


file(GLOB VLLM_TCU_SRC
${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

pybind11_add_module(_C ${VLLM_TCU_SRC})

target_link_libraries(
  _C
  PUBLIC
  ${TORCH_LIBRARIES}
)
install(TARGETS _C DESTINATION vllm_tcu)


