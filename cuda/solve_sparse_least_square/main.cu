#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
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
#define CUSPCHECK(call)                        \
  {                                            \
    const cusparseStatus_t status = call;      \
    assert(status == CUSOLVER_STATUS_SUCCESS); \
  }

int main(int argc, char *argv[]) {
  cusolverSpHandle_t cusolverH = NULL;
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA = NULL;
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;
  double *d_b = NULL;
  double *d_x = NULL;

  /*     | 1.0    0    0    0 |
     A = |   0  2.0    0    0 |
         |   0    0  3.0    0 |
         | 0.1  0.1  0.1  4.0 | */
  const int m = 4;                                     // rows
  const int nnzA = 7;                                  // # of non-zero elements
  const int csrRowPtrA[m + 1] = {0, 1, 2, 3, 7};       // start of every row
  const int csrColIndA[nnzA] = {0, 1, 2, 0, 1, 2, 3};  // column index
  const double csrValA[nnzA] = {1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};

  const double b[m] = {1.0, 1.0, 1.0, 1.0};

  CUSLVCHECK(cusolverSpCreate(&cusolverH));
  CUSPCHECK(cusparseCreateMatDescr(&descrA));
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);  // base-1
  CUSLVCHECK(cusolverSpCreateCsrqrInfo(&info));

  CUDACHECK(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
  CUDACHECK(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  CUDACHECK(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (m + 1)));
  CUDACHECK(cudaMalloc((void **)&d_b, sizeof(double) * m));
  CUDACHECK(cudaMalloc((void **)&d_x, sizeof(double) * m));

  CUDACHECK(cudaMemcpy(d_csrValA, csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_b, b, sizeof(double) * m, cudaMemcpyHostToDevice));

  double tol = 1e-6;
  int reorder = 1;
  int singularity;
  CUSLVCHECK(cusolverSpDcsrlsvqr(cusolverH, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                       d_csrColIndA, d_b, tol, reorder, d_x, &singularity));

  double res_x[4];
  CUDACHECK(cudaMemcpy(res_x, d_x, sizeof(double) * m, cudaMemcpyDeviceToHost));
  for(int i = 0; i < 4; i++) {
    std::cout << res_x[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}

