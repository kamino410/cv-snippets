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

void printMatrix(int m, int n, const double *A, int lda) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) { std::cout << A[row + col * lda] << ",\t"; }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  cusolverSpHandle_t cusolverH = NULL;
  // GPU does batch QR
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA = NULL;
  // GPU does batch QR
  // d_A is CSR format, d_csrValA is of size nnzA*batchSize
  // d_x is a matrix of size batchSize * m
  // d_b is a matrix of size batchSize * m
  int *d_csrRowPtrA = NULL;
  int *d_csrColIndA = NULL;
  double *d_csrValA = NULL;
  double *d_b = NULL;  // batchSize * m
  double *d_x = NULL;  // batchSize * m
  size_t size_qr = 0;
  size_t size_internal = 0;
  void *buffer_qr = NULL;  // working space for numerical factorization

  const int m = 4;
  const int nnzA = 7;
  const int csrRowPtrA[m + 1] = {1, 2, 3, 4, 8};
  const int csrColIndA[nnzA] = {1, 2, 3, 1, 2, 3, 4};
  const double csrValA[nnzA] = {1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
  const double b[m] = {1.0, 1.0, 1.0, 1.0};
  const int batchSize = 17;
  double *csrValABatch = (double *)malloc(sizeof(double) * nnzA * batchSize);
  double *bBatch = (double *)malloc(sizeof(double) * m * batchSize);
  double *xBatch = (double *)malloc(sizeof(double) * m * batchSize);
  assert(NULL != csrValABatch);
  assert(NULL != bBatch);
  assert(NULL != xBatch);

  // step 1: prepare Aj and bj on host
  //  Aj is a small perturbation of A
  //  bj is a small perturbation of b
  //  csrValABatch = [A0, A1, A2, ...]
  //  bBatch = [b0, b1, b2, ...]
  for (int colidx = 0; colidx < nnzA; colidx++) {
    double Areg = csrValA[colidx];
    for (int batchId = 0; batchId < batchSize; batchId++) {
      double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
      csrValABatch[batchId * nnzA + colidx] = Areg + eps;
    }
  }
  for (int j = 0; j < m; j++) {
    double breg = b[j];
    for (int batchId = 0; batchId < batchSize; batchId++) {
      double eps = ((double)((rand() % 100) + 1)) * 1.e-4;
      bBatch[batchId * m + j] = breg + eps;
    }
  }

  // step 2: create cusolver handle, qr info and matrix descriptor
  CUSLVCHECK(cusolverSpCreate(&cusolverH));
  CUSPCHECK(cusparseCreateMatDescr(&descrA));
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);  // base-1
  CUSLVCHECK(cusolverSpCreateCsrqrInfo(&info));

  // step 3: copy Aj and bj to device
  CUDACHECK(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA * batchSize));
  CUDACHECK(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
  CUDACHECK(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (m + 1)));
  CUDACHECK(cudaMalloc((void **)&d_b, sizeof(double) * m * batchSize));
  CUDACHECK(cudaMalloc((void **)&d_x, sizeof(double) * m * batchSize));

  CUDACHECK(cudaMemcpy(d_csrValA, csrValABatch, sizeof(double) * nnzA * batchSize,
                       cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(d_b, bBatch, sizeof(double) * m * batchSize, cudaMemcpyHostToDevice));

  // step 4: symbolic analysis
  CUSLVCHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                             d_csrColIndA, info));

  // step 5: prepare working space
  CUSLVCHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                               d_csrRowPtrA, d_csrColIndA, batchSize, info,
                                               &size_internal, &size_qr));
  printf("numerical factorization needs internal data %lld bytes\n", (long long)size_internal);
  printf("numerical factorization needs working space %lld bytes\n", (long long)size_qr);
  CUDACHECK(cudaMalloc((void **)&buffer_qr, size_qr));

  // step 6: numerical factorization
  // assume device memory is big enough to compute all matrices.
  CUSLVCHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                       d_csrColIndA, d_b, d_x, batchSize, info, buffer_qr));

  // step 7: check residual
  // xBatch = [x0, x1, x2, ...]
  CUDACHECK(cudaMemcpy(xBatch, d_x, sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost));
  const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA)) ? 1 : 0;
  for (int batchId = 0; batchId < batchSize; batchId++) {
    // measure |bj - Aj*xj|
    double *csrValAj = csrValABatch + batchId * nnzA;
    double *xj = xBatch + batchId * m;
    double *bj = bBatch + batchId * m;
    // sup| bj - Aj*xj|
    double sup_res = 0;
    for (int row = 0; row < m; row++) {
      const int start = csrRowPtrA[row] - baseA;
      const int end = csrRowPtrA[row + 1] - baseA;
      double Ax = 0.0;  // Aj(row,:)*xj
      for (int colidx = start; colidx < end; colidx++) {
        const int col = csrColIndA[colidx] - baseA;
        const double Areg = csrValAj[colidx];
        const double xreg = xj[col];
        Ax = Ax + Areg * xreg;
      }
      double r = bj[row] - Ax;
      sup_res = (sup_res > fabs(r)) ? sup_res : fabs(r);
    }
    printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
  }

  for (int batchId = 0; batchId < batchSize; batchId++) {
    double *xj = xBatch + batchId * m;
    for (int row = 0; row < m; row++) { printf("x%d[%d] = %E\n", batchId, row, xj[row]); }
    printf("\n");
  }

  return 0;
}

