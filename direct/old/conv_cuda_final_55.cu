#include "helpers.h"
__constant__ float ckernel[225];

__global__ void conv_cuda(float *input, float *output, int width, int height,
    float *kernel, int channels, int k_width,
    int kernels) {

int k = blockIdx.z;
int j = threadIdx.x + blockIdx.x * blockDim.x;
int i = threadIdx.y + blockIdx.y * blockDim.y;
int output_idx = i * width * kernels + j * kernels + k;
extern __shared__ float sdata[];
int smem_2d_size = (blockDim.x + 2 * k_width) * (blockDim.y + 2 * k_width);

if (threadIdx.y < k_width) {
// Top Left
if (threadIdx.x < k_width) {
int smem_x = threadIdx.x;
int smem_y = threadIdx.y;
int gmem_x = blockIdx.x * blockDim.x + threadIdx.x - k_width;
int gmem_y = blockIdx.y * blockDim.y + threadIdx.y - k_width;
// Top Right
int smem_x1 = smem_x + blockDim.x + k_width;
int smem_y1 = smem_y;
int gmem_x1 = gmem_x + blockDim.x + k_width;
int gmem_y1 = gmem_y;
for (int c = 0; c < channels; c++) {
int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
int smem_index =
(smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;
sdata[smem_index] = (gmem_x < 0 || gmem_y < 0) ? 0 : input[gmem_index];
// Top right
int gmem_index1 = gmem_x1 * channels + gmem_y1 * width * channels + c;
int smem_index1 =
(smem_y1 * (blockDim.x + 2 * k_width) + smem_x1) + c * smem_2d_size;
sdata[smem_index1] =
(gmem_x1 >= width || gmem_y1 < 0) ? 0 : input[gmem_index1];
}
}

// Top Overhang
int smem_x = threadIdx.x + k_width;
int smem_y = threadIdx.y;
int gmem_x = blockIdx.x * blockDim.x + threadIdx.x;
int gmem_y = blockIdx.y * blockDim.y + threadIdx.y - k_width;
// Left Overhang
int smem_x1 = threadIdx.y;
int smem_y1 = threadIdx.x + k_width;
int gmem_x1 = blockIdx.x * blockDim.x + threadIdx.y - k_width;
int gmem_y1 = blockIdx.y * blockDim.y + threadIdx.x;
// // Right Overhang
// int smem_x2 = blockDim.x +  k_width + smem_x1;
// int smem_y2 = smem_y1;
// int gmem_x2 = gmem_x1 + blockDim.x + k_width ;
// int gmem_y2 = gmem_y1;
// // Bottom Overhang
// int smem_x3 = smem_x;
// int smem_y3 = blockDim.y +  k_width + smem_y;
// int gmem_x3 = gmem_x;
// int gmem_y3 = gmem_y + blockDim.y + k_width;

for (int c = 0; c < channels; c++) {
// Assign Top values
int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
int smem_index =
(smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;
sdata[smem_index] = (gmem_y < 0) ? 0 : input[gmem_index];
// Assign Left value
int gmem_index1 = gmem_x1 * channels + gmem_y1 * width * channels + c;
int smem_index1 =
(smem_y1 * (blockDim.x + 2 * k_width) + smem_x1) + c * smem_2d_size;
sdata[smem_index1] = (gmem_x < 0) ? 0 : input[gmem_index1];
// //Assign Right 
// int gmem_index2 = gmem_x2 * channels + gmem_y2 * width * channels + c;
// int smem_index2 =
//     (smem_y2 * (blockDim.x + 2 * k_width) + smem_x2) + c * smem_2d_size;
// sdata[smem_index2] = (gmem_x3 >= width) ? 0 : input[gmem_index2];
// //Assign Bottom
// int gmem_index3 = gmem_x3 * channels + gmem_y3 * width * channels + c;
// int smem_index3 =
//     (smem_y3 * (blockDim.x + 2 * k_width) + smem_x3) + c * smem_2d_size;
// sdata[smem_index3] = (gmem_y3 >= height) ? 0 : input[gmem_index3];
// //Assign Bottom
}
}

// Bottom Left
if (threadIdx.x < k_width && threadIdx.y >= blockDim.y - k_width) {
int smem_x = threadIdx.x;
int smem_y = threadIdx.y + 2 * k_width;
int gmem_x = blockIdx.x * blockDim.x + threadIdx.x - k_width;
int gmem_y = blockIdx.y * blockDim.y + threadIdx.y + k_width;
int smem_x1 = smem_x + blockDim.x + k_width;
int smem_y1 = smem_y;
int gmem_x1 = gmem_x + blockDim.x + k_width;
int gmem_y1 = gmem_y;
for (int c = 0; c < channels; c++) {
int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
int smem_index =
(smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;

sdata[smem_index] =
(gmem_x < 0 || gmem_y >= height) ? 0 : input[gmem_index];
int gmem_index1 = gmem_x1 * channels + gmem_y1 * width * channels + c;
int smem_index1 =
(smem_y1 * (blockDim.x + 2 * k_width) + smem_x1) + c * smem_2d_size;
sdata[smem_index1] =
(gmem_x1 >= width || gmem_y1 >= height) ? 0 : input[gmem_index1];
}
}
//   // Bottom Right
//   if (threadIdx.x >= blockDim.x - k_width &&
//     threadIdx.y >= blockDim.y - k_width) {
//   int smem_x = threadIdx.x + 2 * k_width;
//   int smem_y = threadIdx.y + 2 * k_width;
//   int gmem_x = blockIdx.x * blockDim.x + threadIdx.x + k_width;
//   int gmem_y = blockIdx.y * blockDim.y + threadIdx.y + k_width;
//   for (int c = 0; c < channels; c++) {
//     int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
//     int smem_index =
//         (smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;
//     sdata[smem_index] =
//         (gmem_x >= width || gmem_y >= height) ? 0 : input[gmem_index];
//   }
// }

// Bottom
if (threadIdx.y >= blockDim.y - k_width) {
// Indexes for bottom padding
int smem_x = threadIdx.x + k_width;
int smem_y = threadIdx.y + 2 * k_width;
int gmem_x = blockIdx.x * blockDim.x + threadIdx.x;
int gmem_y = blockIdx.y * blockDim.y + threadIdx.y + k_width;
// Indexes for right side
int smem_x1 = threadIdx.y + 2 * k_width;
int smem_y1 = threadIdx.x + k_width;
int gmem_x1 = blockIdx.x * blockDim.x + threadIdx.y + k_width;
int gmem_y1 = blockIdx.y * blockDim.y + threadIdx.x;

for (int c = 0; c < channels; c++) {
int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
int smem_index =
(smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;
sdata[smem_index] = (gmem_y >= height) ? 0 : input[gmem_index];
int gmem_index1 = gmem_x1 * channels + gmem_y1 * width * channels + c;
int smem_index1 =
(smem_y1 * (blockDim.x + 2 * k_width) + smem_x1) + c * smem_2d_size;
sdata[smem_index1] = (gmem_x1 >= width) ? 0 : input[gmem_index1];
}
}

// Copy the block data
int smem_x = threadIdx.x + k_width;
int smem_y = threadIdx.y + k_width;
int gmem_x = blockIdx.x * blockDim.x + threadIdx.x;
int gmem_y = blockIdx.y * blockDim.y + threadIdx.y;
for (int c = 0; c < channels; c++) {
int gmem_index = gmem_x * channels + gmem_y * width * channels + c;
int smem_index =
(smem_y * (blockDim.x + 2 * k_width) + smem_x) + c * smem_2d_size;
sdata[smem_index] =
(gmem_x >= width || gmem_y >= height) ? 0 : input[gmem_index];
}

__syncthreads();

if (i >= height || j >= width) {
return;
}

float tmp_output = 0;

for (int c = 0; c < channels; c++) {
for (int k_i = 0; k_i <= 2 * k_width; k_i++) {
for (int k_j = 0; k_j <= 2 * k_width; k_j++) {

smem_x = threadIdx.x + k_j;
smem_y = threadIdx.y + k_i;
int smem_index =
c * smem_2d_size + smem_x + smem_y * (blockDim.x + 2 * k_width);
int kernel_index =
k * channels * (2 * k_width + 1) * (2 * k_width + 1) +
c * (2 * k_width + 1) * (2 * k_width + 1) +
k_i * (2 * k_width + 1) + k_j;
tmp_output += sdata[smem_index] * kernel[kernel_index];
}
}
}

output[output_idx] = tmp_output;

return;
}

int main(int argc, char *argv[]) {
  char *outputfile = (char *)"cuda_out_reorder.png";
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
  int padding = 2;
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
  // float *h_output = new float[output_bytes];
  float *h_output;
  h_output = (float *)malloc(output_bytes);
  float *d_input;
  float *d_output;
  cudaMalloc((void **)&d_input, input_bytes);
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

  const float kernel_template[5][5] = {{1, 1, 1, 1, 1},
                                       {1, 4, 4, 4, 1},
                                       {1, 4, 12, 4, 1},
                                       {1, 4, 4, 4, 1},
                                       {1, 1, 1, 1, 1}};
  float *d_kernel;
  float h_kernel[3][3][5][5];
  int kernel_bytes = 3 * 3 * 5 * 5 * sizeof(float);
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row <5; ++row) {
        for (int column = 0; column < 5; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }
  cudaMalloc((void **)&d_kernel, kernel_bytes);
  cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ckernel, &h_kernel,kernel_bytes);
  int k_size = 5;
  int k_width = 2;

  int smem_size =
      (bx + 2 * k_width) * (by + 2 * k_width) * channels * sizeof(float);
  printf("SMEM size is %d \n", (bx + 2 * k_width) * (by + 2 * k_width));
  //==================================
  // CPU Convolution
  //==================================
  printf("Start conv\n");
  double timeStampA = getTimeStamp();

  conv_cuda<<<grid, block, smem_size>>>(d_input, d_output, width, height,
                                        d_kernel, 3, k_width, kernels);

  cudaDeviceSynchronize();
  double timeStampB = getTimeStamp();

  cudaMemcpy(h_output, d_output, input_bytes, cudaMemcpyDeviceToHost);

  //==================================
  // Collect data
  //==================================

  // Print result
  std::cout << "Total convolution time: " << timeStampB - timeStampA
            << std::endl;
  std::cout << "Save Output to " << outputfile << std::endl;
  save_image(outputfile, h_output, height, width);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_kernel);
  cudaDeviceReset();

  delete[] h_output;
  return 0;
} 