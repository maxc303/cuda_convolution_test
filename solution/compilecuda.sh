nvcc -g conv_cuda_smem_reorder.cu helpers.cu -o conv_smem_reorder -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

nvcc -g conv_cuda_smem_reorder_55.cu helpers.cu -o conv_smem_reorder_55 -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
