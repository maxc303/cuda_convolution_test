nvcc -g conv_cpu.cu helpers.cu -o conv_cpu_test -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
nvcc -g conv_cuda_naive.cu helpers.cu -o conv_cuda_naive -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 
