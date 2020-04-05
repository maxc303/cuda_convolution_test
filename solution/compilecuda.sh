nvcc -g conv_cuda_smem_combine.cu helpers.cu -o conv_smem_combine -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

