#include <stdio.h>
#include <cub/cub.cuh>

void scan_on_device() {
  // Declare, allocate, and initialize device pointers for input and output
  int num_items = 7;
  int *d_in;
  int h_in[] = {8, 6, 7, 5, 3, 0, 9};
  int sz = sizeof(h_in) / sizeof(h_in[0]);
  int *d_out; // e.g., [ , , , , , , ]
  cudaMalloc(&d_in, sz * sizeof(h_in[0]));
  cudaMalloc(&d_out, sz * sizeof(h_in[0]));
  cudaMemcpy(d_in, h_in, sz * sizeof(h_in[0]), cudaMemcpyHostToDevice);

  printf("\nInput:\n");
  for (int i = 0; i < sz; i++)
    printf("%d ", h_in[i]);

  // Determine temporary device storage requirements
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  // d_out s<-- [8, 14, 21, 26, 29, 29, 38]
  cudaMemcpy(h_in, d_out, sz * sizeof(h_in[0]), cudaMemcpyDeviceToHost);

  printf("\nOutput:\n");
  for (int i = 0; i < sz; i++)
    printf("%d ", h_in[i]);
  printf("\n");
}

int main(void) {
  scan_on_device();
  return 0;
}