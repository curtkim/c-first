#include <cublas_v2.h>
#include <iostream>
#include <sys/time.h>
#include <thrust/device_vector.h>

typedef struct {
  struct timeval startTime;
  struct timeval endTime;
} Timer;

void startTime(Timer *timer) { gettimeofday(&(timer->startTime), NULL); }

void stopTime(Timer *timer) { gettimeofday(&(timer->endTime), NULL); }

// C-style indexing
int ci(int row, int column, int nColumns) { return row * nColumns + column; }

float elapsedTime(Timer timer) {
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                  (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}

int main(void) {
  Timer timer;
  int matArow = 1000;    // number of rows in A = m
  int matAcol = 1000;    // number of columns in A = n
  int matBrow = matAcol; // number of rows in B = n
  int matBcol = 1000;    // number of columns in B = k
  int matCrow = matArow; // number of rows in C = m
  int matCcol = matBcol; // number of cplumns in C = n

  // initialize data
  thrust::device_vector<float> A(matArow * matAcol);
  thrust::device_vector<float> B(matBrow * matBcol);
  thrust::device_vector<float> C(matCrow * matCcol);

  for (size_t i = 0; i < matArow; i++) {
    for (size_t j = 0; j < matAcol; j++) {
      A[ci(i, j, matAcol)] = i + j;
    }
  }

  for (size_t i = 0; i < matBrow; i++) {
    for (size_t j = 0; j < matBcol; j++) {
      B[ci(i, j, matBcol)] = i + j;
    }
  }

  for (size_t i = 0; i < matCrow; i++)
    for (size_t j = 0; j < matCcol; j++)
      C[ci(i, j, matCcol)] = 0;

  cublasHandle_t handle;

  /* Initialize CUBLAS */
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization error\n";
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  startTime(&timer);
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matBcol, matArow,
                       matAcol, &alpha, thrust::raw_pointer_cast(&B[0]),
                       matBcol, thrust::raw_pointer_cast(&A[0]), matAcol, &beta,
                       thrust::raw_pointer_cast(&C[0]), matBcol);
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "kernel execution error.\n";
  }

  for (size_t i = 0; i < matCrow; i++) {
    for (size_t j = 0; j < matCcol; j++) {
    }
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "shutdown error\n";
  }

  return 0;
}
