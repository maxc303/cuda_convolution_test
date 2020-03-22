nvcc -g conv_cpu.cu -o conv_cpu_test -lcudart -lcublas -lcudnn -lstdc++ -lm -lopencv_core -lopencv_imgproc -lopencv_highgui 

