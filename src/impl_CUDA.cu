#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#define d(i, j) ((i) * N + (j))
#define f(i, j) ((i) * N / 2 + ((j) / 2))
__global__ void kernel_col1(int N, double *p1, double *p2) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 2 - i % 2;
  if (i < N - 1 && j < N - 1) {
      p1[f(i, j)] = (p2[f(i - 1, j)] + p2[f(i + 1, j)] + p2[f(i, j - 1)] + p2[f(i, j + 1)]) / 4.0f;
  }
}
__global__ void kernel_col2(int N, double *p1, double *p2) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + i % 2 + 1;
  if (i < N - 1 && j < N - 1) {
      p2[f(i, j)] = (p1[f(i - 1, j)] + p1[f(i + 1, j)] + p1[f(i, j - 1)] + p1[f(i, j + 1)]) / 4.0f;
  }
}
extern "C" void impl_CUDA(int N, int step, double *p) {
  cudaSetDevice(0);
  double *p1 = (double*)calloc((N / 2 + 2) * N, sizeof(double));
  double *p2 = (double*)calloc((N / 2 + 2) * N, sizeof(double));
  double *d_p1, *d_p2;
  cudaMalloc(&d_p1, (N / 2 + 2) * N * sizeof(double));
  cudaMalloc(&d_p2, (N / 2 + 2) * N * sizeof(double));
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if((i + j) % 2 == 0) p1[f(i, j)] = p[d(i, j)];
      else p2[f(i, j)] = p[d(i, j)];
    }
  }
  cudaMemcpy(d_p1, p1, (N / 2 + 2) * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2, p2, (N / 2 + 2) * N * sizeof(double), cudaMemcpyHostToDevice);
  dim3 block(16, 16);
  dim3 grid((N - 1) / 2 / block.x + 1, (N - 2) / block.y + 1);
  for(int k = 1; k <= step; k++){
    if(k % 2){
      kernel_col1<<<grid, block>>>(N, d_p1, d_p2);
    }else{
      kernel_col2<<<grid, block>>>(N, d_p1, d_p2);
    }
    cudaDeviceSynchronize();
  }
  cudaMemcpy(p1, d_p1, (N / 2 + 2) * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p2, d_p2, (N / 2 + 2) * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_p1);
  cudaFree(d_p2);
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if((i + j) % 2 == 0) p[d(i, j)] = p1[f(i, j)];
      else p[d(i, j)] = p2[f(i, j)];
    }
  }
  free(p1);
  free(p2);
}