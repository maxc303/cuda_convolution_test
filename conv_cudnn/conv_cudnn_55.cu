#include <cudnn.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
// Check cudnn
#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// Get time function
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

// Save Image function
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
  // Init cudnn
  cudaDeviceReset();

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  char *outputfile = (char *)"cudnn_out_55.png";
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

  // Input Descriptor
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/3,
                                        /*in_channels=*/3,
                                        /*kernel_height=*/5,
                                        /*kernel_width=*/5));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/2,
                                             /*pad_width=*/2,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
      cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
      output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      /*memoryLimitInBytes=*/0, &convolution_algorithm));

  size_t workspace_bytes = 0;

  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
      output_descriptor, convolution_algorithm, &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;

  void *d_workspace;
  cudaMalloc(&d_workspace, workspace_bytes);
  std::cout << "allocate workspace" << std::endl;
  int batch_size;
  int channels;
  int height;
  int width;

  cudnnGetConvolution2dForwardOutputDim(
      convolution_descriptor, input_descriptor, kernel_descriptor, &batch_size,
      &channels, &height, &width);

  int image_bytes = batch_size * channels * height * width * sizeof(float);

  float *d_input;
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

  float *d_output;
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);

  std::cout << "Height and width:" << height << " x " << width << std::endl;
  // Mystery kernel
  const float kernel_template[5][5] = {{1, 1, 1, 1, 1},
                                       {1, 4, 4, 4, 1},
                                       {1, 4, 12, 4, 1},
                                       {1, 4, 4, 4, 1},
                                       {1, 1, 1, 1, 1}};

  float h_kernel[3][3][5][5];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float *d_kernel;
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
  const float alpha = 1, beta = 0;

  std::cout << "Start conv" << std::endl;
  double timeStampA = getTimeStamp();
  checkCUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
      convolution_descriptor, convolution_algorithm, d_workspace,
      workspace_bytes, &beta, output_descriptor, d_output));

  cudaDeviceSynchronize();
  double timeStampB = getTimeStamp();
  float *h_output = new float[image_bytes];
  cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

  // Print result
  std::cout << "Total convolution time: " << timeStampB - timeStampA
            << std::endl;
  std::cout << "Save Output to " << outputfile << std::endl;
  save_image(outputfile, h_output, height, width);

  // Delete
  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
