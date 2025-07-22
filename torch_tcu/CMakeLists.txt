cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(extendBackEnd_FOR_CODE_INDEX_LLM)

# compile options
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

include_directories("/home/leo/.pyenv/versions/anaconda3-2023.09-0/include/python3.11/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/")
# package
#find_package(Torch REQUIRED PATHS "/home/leo/Downloads/dev/libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu/share/cmake/Torch")
find_package(Torch REQUIRED PATHS "/home/leo/Downloads/dev/libtorch-cxx11-abi-shared-with-deps-2.6.0+cp/share/cmake/Torch")

#add_executable(extendBackEnd_FOR_CODE_INDEX_LLM torch_tcu.cpp op_impl_cpu/op_implements.cpp op_implements.h)
# libtorch
#target_link_libraries(extendBackEnd_FOR_CODE_INDEX_LLM ${TORCH_LIBRARIES})
#target_link_libraries(extendBackEnd_FOR_CODE_INDEX_LLM pthread)

add_executable(libtorch_unit_tests op_impl_cpu/op_implements.cpp op_libtorch_tests/op_libtorch_test.cpp)
target_link_libraries(libtorch_unit_tests ${TORCH_LIBRARIES})
