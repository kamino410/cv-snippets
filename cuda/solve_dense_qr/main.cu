#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdlib.h>
#include <iostream>

#define CUDACHECK(call)              \
  {                                  \
    const cudaError_t status = call; \
    assert(status == cudaSuccess);   \
  }
#define CUSLVCHECK(call)                       \
  {                                            \
    const cusolverStatus_t status = call;      \
    assert(status == CUSOLVER_STATUS_SUCCESS); \
  }
#define CUBLASCHECK(call)                    \
  {                                          \
    const cublasStatus_t status = call;      \
    assert(status == CUBLAS_STATUS_SUCCESS); \
  }

void printMatrix(int m, int n, const double *A, int lda) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) { std::cout << A[row + col * lda] << ",\t"; }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  const int m = 3;
  const int lda = m;
  const int ldb = m;
  const int nrhs = 1;  // number of right hand side vectors

  /* | 1 2 3 |
   * A = | 4 5 6 |
   * | 2 1 1 |
   *
   * x = (1 1 1)'
   * b = (6 15 4)'
   */
  double A[lda * m] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
  // double X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
  double B[ldb * nrhs] = {6.0, 15.0, 4.0};
  double XC[ldb * nrhs];  // solution matrix from GPU
  double *d_A = NULL;     // linear memory of GPU
  double *d_tau = NULL;   // linear memory of GPU
  double *d_B = NULL;
  int *devInfo = NULL;  // info in gpu (device copy)
  double *d_work = NULL;
  int lwork = 0;
  int info_gpu = 0;
  const double one = 1;
  std::cout << "A = (matlab base-1)" << std::endl;
  printMatrix(m, m, A, lda);
  std::cout << "=====" << std::endl;
  std::cout << "B = (matlab base-1)" << std::endl;
  printMatrix(m, nrhs, B, ldb);
  std::cout << "=====" << std::endl;
  // step 1: create cusolver/cublas handle
  CUSLVCHECK(cusolverDnCreate(&cusolverH));
  CUBLASCHECK(cublasCreate(&cublasH));

  // step 2: copy A and B to device
  CUDACHECK(cudaMalloc((void **)&d_A, sizeof(double) * lda * m));
  CUDACHECK(cudaMalloc((void **)&d_tau, sizeof(double) * m));
  CUDACHECK(cudaMalloc((void **)&d_B, sizeof(double) * ldb * nrhs));
  CUDACHECK(cudaMalloc((void **)&devInfo, sizeof(int)));
  CUDACHECK(cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice));

  // step 3: query working space of geqrf and ormqr
  CUSLVCHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
  CUDACHECK(cudaMalloc((void **)&d_work, sizeof(double) * lwork));
  // step 4: compute QR factorization
  CUSLVCHECK(cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo));
  CUDACHECK(cudaDeviceSynchronize());
  // check if QR is good or not
  CUDACHECK(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "after geqrf: info_gpu = " << info_gpu << std::endl;
  assert(0 == info_gpu);
  // step 5: compute Q^T*B
  CUSLVCHECK(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau,
                              d_B, ldb, d_work, lwork, devInfo));
  CUDACHECK(cudaDeviceSynchronize());

  // check if QR is good or not
  CUDACHECK(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "after ormqr: info_gpu = " << info_gpu << std::endl;
  assert(0 == info_gpu);
  // step 6: compute x = R \ Q^T*B
  CUBLASCHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                          CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb));
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaMemcpy(XC, d_B, sizeof(double) * ldb * nrhs, cudaMemcpyDeviceToHost));
  std::cout << "X = (matlab base-1)" << std::endl;
  printMatrix(m, nrhs, XC, ldb);
  // free resources
  if (d_A) cudaFree(d_A);
  if (d_tau) cudaFree(d_tau);
  if (d_B) cudaFree(d_B);
  if (devInfo) cudaFree(devInfo);
  if (d_work) cudaFree(d_work);
  if (cublasH) cublasDestroy(cublasH);
  if (cusolverH) cusolverDnDestroy(cusolverH);
  cudaDeviceReset();
  return 0;
}

