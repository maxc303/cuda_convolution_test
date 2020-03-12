cmake_minimum_required(VERSION 2.8.9)
project (conv_cudnn)




find_package(CUDA QUIET REQUIRED)


set(CUDA_HOST_COMPILER 
"usr/bin/clang++-3.8"
 )

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
	-ccbin clang-3.8
    )

# Specify include directories
include_directories(
    /usr/include/opencv
    /nfs/ug/homes-2/c/cheny248/cuda/include
    )

# Specify library paths
link_directories(

	/nfs/ug/homes-2/c/cheny248/cuda/lib64

    )



# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    conv_cudnn_test
    conv_cudnn.cu

    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
	conv_cudnn_test
    -lcudnn
	-lopencv_core 
	-lopencv_imgproc 
    -lopencv_highgui
    )

