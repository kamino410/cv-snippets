#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>

#define CHECK(call)                                                      \
  {                                                                      \
    const cudaError_t error = call;                                      \
    if (error != cudaSuccess) {                                          \
      std::cout << "Error: " << __FILE__ << ":" << __LINE__ << std::endl \
                << cudaGetErrorString(error) << std::endl;               \
      exit(1);                                                           \
    }                                                                    \
  }

int main() {
  cv::Mat img = cv::imread("lenna.png", 0);

  int NX = img.cols;
  int NY = img.rows;
  size_t size = sizeof(cufftDoubleComplex) * NX * NY;

  cufftDoubleComplex *input = (cufftDoubleComplex *)malloc(size);
  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      input[NX * y + x] = make_cuDoubleComplex(img.at<uint8_t>(y, x), 0);
    }
  }

  cufftHandle plan;
  cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z);
  cufftDoubleComplex *idata, *odata;

  CHECK(cudaMalloc((void **)&idata, size));
  CHECK(cudaMalloc((void **)&odata, size));
  CHECK(cudaMemcpy(idata, input, size, cudaMemcpyHostToDevice));

  cufftExecZ2Z(plan, idata, odata, CUFFT_FORWARD);
  cufftDoubleComplex *freq = (cufftDoubleComplex *)malloc(size);
  CHECK(cudaMemcpy(freq, odata, size, cudaMemcpyDeviceToHost));

  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      freq[NX * y + x].x /= (NX * NY);
      freq[NX * y + x].y /= (NX * NY);
    }
  }
  std::cout << freq[0].x << ", " << freq[0].y << std::endl;

  CHECK(cudaMemcpy(idata, freq, size, cudaMemcpyHostToDevice));
  cufftExecZ2Z(plan, idata, odata, CUFFT_INVERSE);

  cufftDoubleComplex *result = (cufftDoubleComplex *)malloc(size);
  CHECK(cudaMemcpy(result, odata, size, cudaMemcpyDeviceToHost));

  cufftDestroy(plan);
  cudaFree(idata);
  cudaFree(odata);

  std::cout << result[0].x << ", " << result[0].y << std::endl;
  cv::Mat res_img(NY, NX, CV_8U, cv::Scalar(0));
  for (int y = 0; y < NY; y++) {
    for (int x = 0; x < NX; x++) {
      float real = result[NX * y + x].x;
      float image = result[NX * y + x].y;
      res_img.at<uint8_t>(y, x) = (int)(real);
    }
  }
  cv::imwrite("test.png", res_img);

  free(input);
  free(freq);
  free(result);
}
