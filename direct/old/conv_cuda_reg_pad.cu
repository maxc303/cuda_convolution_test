#include "helpers.h"

__constant__ float ckernel[81];

//Padding function
__global__ void cuda_padding(float *input, float *input_padded, int pad_size,
                             int width, int height, int width_padded,
                             int height_padded, int channels) {
  int c = blockIdx.z;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int input_idx = i * width * channels + j * channels + c;
  // printf("padding input idx i %d , j %d, c %d \n",i,j,c);
  if (i >= height || j >= width) {
    return;
  }
  int output_idx =
      (i + pad_size) * width_padded * channels + (j + pad_size) * channels + c;
  input_padded[output_idx] = input[input_idx];
  // printf("padded value ");
}

//Padding Convolution
__global__ void conv_pad_cuda(float *input, float *output, int width,
                              int height, int width_padded, int height_padded,
                              float *kernel, int channels, int k_width,
                              int kernels,int pad_size) {

  int k = blockIdx.z;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int output_idx = i * width * kernels + j * kernels + k;
  int input_idx = 0;

  float tmp_output=0;
  extern __shared__ float s_out[];
  if (i >= height || j >= width) {
    return;
  }

  output[output_idx] = 0.0;
  
  for (int c = 0; c < channels; c++) {
    for (int k_i = -k_width; k_i <= k_width; k_i++) {
      for (int k_j = -k_width; k_j <= k_width; k_j++) {
        input_idx = (i + k_i + pad_size) * width_padded * channels +
                    (j + k_j + pad_size) * channels + c;
        int kernel_index =
            k * channels * (2 * k_width + 1) * (2 * k_width + 1) +
            c * (2 * k_width + 1) * (2 * k_width + 1) +
            (k_i + k_width) * (2 * k_width + 1) + k_j + k_width;
        //output[output_idx] += input[input_idx] * ckernel[kernel_index];
        tmp_output += input[input_idx] * ckernel[kernel_index];
        // h_kernel[k][c][k_i + k_width][k_j + k_width];
        
      }
    }
  }
  //Write SMEM to output
  output[output_idx]= tmp_output;

  return;
}

int main(int argc, char *argv[]) {
  char *outputfile = (char *)"cuda_reg_out.png";
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
  int padded_bytes = width_padded * height_padded * channels * sizeof(float);
  float *h_input = (float *)image.data;

  // float *h_output = new float[output_bytes];
  float *h_output;
  h_output = (float *)malloc(output_bytes);
  float *d_input;
  float *d_pad_input;
  float *d_output;
  cudaMalloc((void **)&d_input, input_bytes);
  cudaMalloc((void **)&d_pad_input, padded_bytes);
  cudaMalloc((void **)&d_output, output_bytes);
  cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

  // invoke Kernel
  int bx = 32;
  int by = 32;
  dim3 block(bx, by); // you will want to configure this
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
            3);
  printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d} threads.\n", grid.x,
         grid.y, grid.z, block.x, block.y);

  //==================================
  // Define Kernel data
  //==================================
  // Mystery kernel

  const float kernel_template[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
  float *d_kernel;
  float h_kernel[3][3][3][3];
  int kernel_bytes = 3 * 3 * 3 * 3 * sizeof(float);
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  // //Copy kernel to global mem
  // cudaMalloc( (void **) &d_kernel, kernel_bytes ) ;
  // cudaMemcpy( d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
  // Copy kernek to cmem
  cudaMemcpyToSymbol(ckernel, &h_kernel, kernel_bytes);

  int k_size = 3;
  int k_width = (k_size - 1) / 2;
  //==================================
  // CPU Convolution
  //==================================
  printf("Start conv\n");
  double timeStampA = getTimeStamp();

  cuda_padding<<<grid, block>>>(d_input, d_pad_input, padding, width, height,
                                width_padded, height_padded, channels);

                                cudaDeviceSynchronize();
                                double timeStampC = getTimeStamp();
  conv_pad_cuda<<<grid, block, bx*by*sizeof(float)>>>(d_pad_input, d_output, width, height, width_padded,
                             height_padded, ckernel, 3, k_width, kernels, padding);
                             cudaDeviceSynchronize();
  double timeStampB = getTimeStamp();

  cudaMemcpy(h_output, d_output, input_bytes, cudaMemcpyDeviceToHost);

  //==================================
  // Collect data
  //==================================

  // Print result
  std::cout << "Total convolution time: " << timeStampB - timeStampA
            << std::endl;
            std::cout << "Padding time: " << timeStampC - timeStampA
            << std::endl;
  std::cout << "Save Output to " << outputfile << std::endl;
  save_image(outputfile, h_output, height, width);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_pad_input);
  cudaFree(d_kernel);
  cudaDeviceReset();

  return 0;
}