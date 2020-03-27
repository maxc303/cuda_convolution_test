#include "helpers.h"

int main(int argc, char *argv[]) {
  char *outputfile = (char *)"cpu_out.png";
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
  int output_bytes = channels * height * width * sizeof(float);
  std::cout << "Padded dims is " << height_padded << "x" << width_padded
            << std::endl;

 float *h_input = (float *)image.data;
  float *h_output = new float[output_bytes];

  //==================================
  // Define Kernel data
  //==================================
  // Mystery kernel

  const float kernel_template[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  int k_size = 3;
  int k_width = (k_size - 1) / 2;
  //==================================
  // CPU Convolution 
  //==================================

  int input_idx = 0;
  int output_idx = 0;
  double timeStampA = getTimeStamp();
  for (int k = 0; k < kernels; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        output_idx = i*width*kernels + j*kernels + k;
        h_output[output_idx] = 0.0;
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
 

                h_output[output_idx] +=
                    h_input[input_idx] *
                    h_kernel[k][c][k_i + k_width][k_j + k_width];

              }
            }
          }
        }
      }
    }
  }

  //==================================
  // Collect data
  //==================================
  double timeStampB = getTimeStamp();

  // Print result
  std::cout << "Total convolution time: " << timeStampB - timeStampA
            << std::endl;
  std::cout << "Save Output to " << outputfile << std::endl;
  save_image(outputfile, h_output, height, width);

  return 0;
}