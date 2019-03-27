#pragma once

template <typename T>
__global__ void SumMatrix(T* A, T* B, T* C, int nx, int ny) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < nx && y < ny) {
    int idx = y * nx + x;
    C[idx] = A[idx] + B[idx];
  }
}

template <typename T>
__global__ void reduce(T* g_idata, T* g_odata, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n) return;

  T* idata = g_idata + blockIdx.x * blockDim.x;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
