#pragma once

#include <map>
#include "paddle/fluid/platform/enforce.h"
#include "time.h"
#include <sstream>

__device__ inline void VecAddImpl(float* a, float* b, float* c, int idx) {
  c[idx] = a[idx] + b[idx];
}

__device__ inline void VecDotImpl(float* a, float* b, float* c, int idx) {
  c[idx] = a[idx] * b[idx];
}

__global__ void VecAdd(float* A, float* B, float* C, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= N) return;

  VecAddImpl(A, B, C, idx);
}

__global__ void VecDot(float* A, float* B, float* C, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= N) return;

  for (int i = 0; i < 1000; i++) VecDotImpl(A, B, C, idx);
}

template <typename T>
T* CreateVec(int N) {
  float* res;
  PADDLE_ENFORCE(cudaMalloc(&res, N * sizeof(T)));
  return res;
}

template <typename T>
void RandVec(float* arr, int N) {
  for (int i = 0; i < N; i++) {
    arr[i] = rand() * 1. / RAND_MAX;
  }
}

template <typename T>
std::pair<T*, T*> CreateVec1(int N) {
  std::pair<T*, T*> res;
  res.first = new T[N];
  PADDLE_ENFORCE(cudaMalloc(&res.second, N * sizeof(T)));
  return res;
}

template <typename T>
struct HdMem {
  T* d;
  T* h;
  int N;

  HdMem(int N) : N(N) {
    auto res = CreateVec1<T>(N);
    d = res.second;
    h = res.first;
    cudaMemset(d, 0, sizeof(T) * N);
  }

  void ToDevice() {
    PADDLE_ENFORCE(cudaMemcpy(d, h, sizeof(T) * N, cudaMemcpyHostToDevice));
  }

  void ToHost() {
    PADDLE_ENFORCE(cudaMemcpy(h, d, sizeof(T) * N, cudaMemcpyDeviceToHost));
  }
};
