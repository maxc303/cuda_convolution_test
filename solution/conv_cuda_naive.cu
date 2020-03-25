#include "helpers.h"

__global__ void conv_cuda( float *input, float *output, int width, int height, float *kernel, int channels,int k_width,int kernels ){
        int k = blockIdx.z; 
        int j = threadIdx.x + blockIdx.x*blockDim.x ;
        int i = threadIdx.y + blockIdx.y*blockDim.y ;
        int output_idx = i*width*kernels + j*kernels + k;
        int input_idx = 0;
        printf("output index %d\n",output_idx);
        output[output_idx] = 0.0;
        // std::cout<< "Looping index "<< output_idx << std::endl;
        // Channel loop
        // Kernel loop
        for (int c = 0; c < channels; c++) {
          for (int k_i = -k_width; k_i <= k_width; k_i++) {
            for (int k_j = -k_width; k_j <= k_width; k_j++) {
              if (i + k_i >= 0 && i + k_i < height && j + k_j >= 0 &&
                  j + k_j < width) {
                input_idx =
                    c + (j + k_j)*channels + (i + k_i)*channels * width;
                int kernel_index = k*channels*(2*k_width+1)^2 + c*(2*k_width+1)^2 + (k_i + k_width)* (2*k_width+1)+k_j + k_width;
                output[output_idx] +=
                    input[input_idx] * kernel[kernel_index];
                    //h_kernel[k][c][k_i + k_width][k_j + k_width];

              }
            }
          }
        }       

    return;
 }


int main(int argc, char *argv[]) {
  char *outputfile = (char *)"cuda_out.png";
  // Check input image name
  if (argc < 2) {
    std::cout << "No file input" << std::endl;
    return 0;
  }
  //
  // Check if the filename is valid
  char *filename = argv[1];
  std::cout << argv[1] << std::endl;
  // Load Image
  cv::Mat image;
  image = load_image(filename);
  if (image.empty()) {
    std::cout << "File not exist" << std::endl;
    return 0;
  }

  double timeStampA = getTimeStamp();

  //==================================
  // Define I/O sizes
  //==================================
  int padding = 1;
  int channels = 3;
  int height = image.rows;
  int width = image.cols;
  int kernels = 3;

  std::cout << "Image dims (HxW)is " << height << "x" << width << std::endl;
  int height_padded = height + 2 * padding;
  int width_padded = width + 2 * padding;
  int input_bytes = channels * height * width * sizeof(float);
  int output_bytes = channels * height * width * sizeof(float);
  std::cout << "Padded dims is " << height_padded << "x" << width_padded
            << std::endl;

  float *h_input = (float *)image.data;
  float *h_output = new float[output_bytes];
  float *d_input;
  float *d_output;
  cudaMalloc( (void **) &d_input, input_bytes) ; 
  cudaMalloc( (void **) &d_output, output_bytes) ; 
  cudaMemcpy( d_input, h_input, input_bytes, cudaMemcpyHostToDevice);


      // invoke Kernel
    int bx =32;
    int by =32;
    dim3 block( bx, by ,3 ) ; // you will want to configure this
    dim3 grid( (width + block.x-1)/block.x, (height + block.y-1)/block.y) ;
    printf("Grid : {%d, %d} blocks. Blocks : {%d, %d} threads.\n", grid.x, grid.y, block.x, block.y);


  //==================================
  // Define Kernel data
  //==================================
  // Mystery kernel

  const float kernel_template[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  float *d_kernel;
  float h_kernel[3][3][3][3];
  int kernel_bytes = 3*3*3*3*sizeof(float);
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }
  cudaMalloc( (void **) &d_kernel, kernel_bytes ) ;
  cudaMemcpy( d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

  int k_size = 3;
  int k_width = (k_size - 1) / 2;
  //==================================
  // CPU Convolution 
  //==================================
  printf("Start conv\n");

  conv_cuda<<<grid, block>>>( d_input, d_output, width, height, d_kernel, 3,k_width,kernels );
  cudaMemcpy( h_output, d_output, input_bytes, cudaMemcpyDeviceToHost);

  //==================================
  // Collect data
  //==================================
  double timeStampB = getTimeStamp();

  // Print result
  std::cout << "Total convolution time: " << timeStampB - timeStampA
            << std::endl;
  std::cout << "Save Output to " << outputfile << std::endl;
  save_image(outputfile, h_output, height, width);


  cudaFree( d_input ) ;
  cudaFree( d_output ) ;
  cudaFree( d_kernel ) ;
  cudaDeviceReset() ;

  delete[] h_output;
  return 0;
}