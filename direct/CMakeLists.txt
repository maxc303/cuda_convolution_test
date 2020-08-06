cmake_minimum_required(VERSION 2.8.9)
project (conv_direct)

find_package(CUDA QUIET REQUIRED)



add_library(helpers STATIC helpers.cpp)

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
# For linking ...
# Specify target & libraries to link it with
link_libraries(
    helpers
    -lcudnn
    -lopencv_core 
    -lopencv_imgproc 
    -lopencv_highgui
)

#CPU test
cuda_add_executable(conv_cpu_test conv_cpu.cu)

#Naive CUDA test
cuda_add_executable(conv_cuda_naive conv_cuda_naive.cu)

#Register Output
cuda_add_executable(conv_cuda_reg_out conv_cuda_reg_nopad.cu)
