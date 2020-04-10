#nvcc -g conv_smem_comb_overhang.cu helpers.cu -o conv_smem_comb_overhang -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_cuda_smem_combine.cu helpers.cu -o conv_smem_combine -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

nvcc -g conv_reg_nopad.cu helpers.cu -o conv_reg_out -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

