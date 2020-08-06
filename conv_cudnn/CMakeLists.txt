cmake_minimum_required(VERSION 2.8.9)
project (conv_cudnn)

find_package(CUDA QUIET REQUIRED)

#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
# For linking ...
# Specify target & libraries to link it with
link_libraries(
    -lcudnn
    -lopencv_core 
    -lopencv_imgproc 
    -lopencv_highgui
)


cuda_add_executable(conv_cudnn_3x3  conv_cudnn.cu )
cuda_add_executable(conv_cudnn_5x5  conv_cudnn_55.cu )



