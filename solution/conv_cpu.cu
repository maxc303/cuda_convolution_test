#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

double getTimeStamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

// Load Image function
cv::Mat load_image(const char *image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  return image;
}
void save_image(const char *output_filename, float *buffer, int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image, output_image,
                /*threshold=*/0,
                /*maxval=*/0, cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);

  cv::imwrite(output_filename, output_image);
}
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
  int output_bytes = channels * height * width * sizeof(float);
  int input_bytes = channels * height * width * sizeof(float);
  std::cout << "Padded dims is " << height_padded << "x" << width_padded
            << std::endl;

  float *h_input = (float *)image.data;
  float *h_output = new float[output_bytes];

  //==================================
  // Write Image to vector
  //==================================
  // int counter =0;
  // for (int z = 0; z < channels; z++){
  //     for (int i = 0; i < width_padded; i++){
  //         for (int j = 0; j < height_padded; j++){
  //             if(i==0 || j ==0 ||i==width_padded-1 ||j ==height_padded-1){
  //                 h_input[counter] = 0;
  //             }else{
  //                 h_input[counter] = image.at<cv::Vec3f>(i-1,j-1)[z] ;
  //             }
  //             counter++;
  //         }
  //     }
  // }

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
  // Convolution data
  //==================================

  int input_idx = 0;
  int output_idx = 0;
  for (int k = 0; k < kernels; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        h_output[output_idx] = 0;
        // std::cout<< "Looping index "<< output_idx << std::endl;
        // Channel loop

        // Kernel loop
        for (int k_i = -k_width; k_i < k_width; k_i++) {
          for (int k_j = -k_width; k_j < k_width; k_j++) {
            if (i + k_i > 0 && i + k_i < height && j + k_j > 0 &&
                j + k_j < width) {
              for (int c = 0; c < 3; c++) {
                input_idx =
                    c * (width * height) + (j + k_j) + (i + k_i) * width;
                h_output[output_idx] +=
                    h_input[input_idx] *
                    kernel_template[k_i + k_width][k_j + k_width];
              }
            }
          }
        }

        output_idx++;
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