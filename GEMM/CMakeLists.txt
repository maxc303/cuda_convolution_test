cmake_minimum_required(VERSION 2.8.9)
project (conv_direct)

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


