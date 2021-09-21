cmake_minimum_required(VERSION 3.0.0)
project(NVIDIA_FFT VERSION 0.1.0 LANGUAGES CUDA CXX) 

find_package(CUDAToolkit REQUIRED)

add_executable(NVIDIA_FFT main.cu)
target_link_libraries(NVIDIA_FFT PUBLIC CUDA::cufft fftw3f_omp fftw3f)
set_target_properties(NVIDIA_FFT PROPERTIES CUDA_ARCHITECTURES "86")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
