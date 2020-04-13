#nvcc -g conv_smem_comb_overhang.cu helpers.cu -o conv_smem_comb_overhang -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_cuda_smem_combine.cu helpers.cu -o conv_smem_combine -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

#nvcc -g conv_cuda_smem_nopad.cu helpers.cu -o conv_cuda_smem_nopad -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_cuda_smem_reorder.cu helpers.cu -o conv_cuda_smem_reorder -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_cuda_smem_combine.cu helpers.cu -o conv_cuda_smem_combine -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_smem_comb_overhang.cu helpers.cu -o conv_smem_comb_overhang -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
#nvcc -g conv_comb_overhang_hwc.cu helpers.cu -o conv_comb_overhang_hwc -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
nvcc -g conv_cuda_combine_2d.cu helpers.cu -o conv_cuda_combine_2d -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

