#include <cuda_profiler_api.h>
#include <gtest/gtest.h>
#include "basic.h"
#include "math.h"
#include "time.h"

TEST(basic, test0) {
  cudaProfilerStop();
  const int dim = 100000;
  float* aD = CreateVec<float>(dim);
  float* bD = CreateVec<float>(dim);
  float* cD = CreateVec<float>(dim);

  float* a = new float[dim];
  float* b = new float[dim];
  float* c = new float[dim];

  RandVec<float>(a, dim);
  RandVec<float>(b, dim);
  RandVec<float>(c, dim);

  const int nbyte = dim * sizeof(float);
  cudaMemcpy(aD, a, nbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(bD, b, nbyte, cudaMemcpyHostToDevice);
  cudaMemcpy(cD, c, nbyte, cudaMemcpyHostToDevice);

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};

  cudaProfilerStart();
  VecAdd<<<block_num, threadnum>>>(aD, bD, cD, dim);

  cudaMemcpy(c, cD, nbyte, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaProfilerStop();
}

TEST(basic, stream) {
  cudaStream_t stream0, stream1, stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int dim = 1000;
  HdMem<float> A(dim);
  HdMem<float> B(dim);
  HdMem<float> C(dim);

  A.ToDevice();
  B.ToDevice();
  C.ToDevice();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};

  cudaProfilerStart();
  VecDot<<<block_num, threadnum, 0, stream0>>>(A.d, B.d, C.d, dim);
  VecDot<<<block_num, threadnum, 0, stream1>>>(A.d, B.d, C.d, dim);
  VecDot<<<block_num, threadnum, 0, stream2>>>(A.d, B.d, C.d, dim);
  cudaProfilerStop();
}

TEST(basic, event) {
  cudaStream_t stream0, stream1, stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaEvent_t event0, event1;
  cudaEventCreate(&event0);
  cudaEventCreate(&event1);

  const int dim = 1000;
  HdMem<float> A(dim);
  HdMem<float> B(dim);
  HdMem<float> C(dim);

  A.ToDevice();
  B.ToDevice();
  C.ToDevice();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};

  cudaProfilerStart();
  VecDot<<<block_num, threadnum, 0, stream0>>>(A.d, B.d, C.d, dim);
  cudaEventRecord(event0, stream0);
  VecDot<<<block_num, threadnum, 0, stream1>>>(A.d, B.d, C.d, dim);
  cudaEventRecord(event1, stream1);

  // make this kernel wait for the previous two kernel finish.
  cudaStreamWaitEvent(stream2, event0, 0);
  cudaStreamWaitEvent(stream2, event1, 0);
  VecDot<<<block_num, threadnum, 0, stream2>>>(A.d, B.d, C.d, dim);

  cudaProfilerStop();
}

TEST(basic, pinned_memory) {
  const int dim = 10000;
  float* A;
  float* dA;
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // malloc pined memory
  cudaMallocHost(&A, dim * sizeof(float), 0);
  memset(A, 0, dim * sizeof(float));

  // malloc cuda memory
  dA = CreateVec<float>(dim);

  // prepare for a compute kernel
  HdMem<float> a(dim);
  HdMem<float> b(dim);
  HdMem<float> c(dim);
  a.ToDevice();
  b.ToDevice();
  c.ToDevice();

  cudaProfilerStart();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};
  // begin parallel
  VecDot<<<threadnum, block_num, 0, stream1>>>(a.d, b.d, c.d, dim);

  for (int i = 0; i < 10; i++) {
    cudaMemcpyAsync(dA, A, sizeof(float) * dim, cudaMemcpyHostToDevice,
                    stream0);
  }

  cudaDeviceSynchronize();
}

TEST(basic, pagable_memory) {
  const int dim = 10000;
  float* A;
  float* dA;
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // malloc pined memory
  A = new float[dim];
  memset(A, 0, dim * sizeof(float));

  // malloc cuda memory
  dA = CreateVec<float>(dim);

  // prepare for a compute kernel
  HdMem<float> a(dim);
  HdMem<float> b(dim);
  HdMem<float> c(dim);
  a.ToDevice();
  b.ToDevice();
  c.ToDevice();

  cudaProfilerStart();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};
  // begin parallel
  VecDot<<<threadnum, block_num, 0, stream1>>>(a.d, b.d, c.d, dim);

  for (int i = 0; i < 10; i++) {
    cudaMemcpyAsync(dA, A, sizeof(float) * dim, cudaMemcpyHostToDevice,
                    stream0);
  }

  cudaDeviceSynchronize();
}

TEST(basic, huge_stream_num) {
  const int kStreamNum = 4000;
  cudaStream_t streams[kStreamNum];
  for (int i = 0; i < kStreamNum; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int dim = 1000;
  HdMem<float> A(dim);
  HdMem<float> B(dim);
  HdMem<float> C(dim);

  RandVec<float>(A.h, dim);
  RandVec<float>(B.h, dim);
  RandVec<float>(C.h, dim);

  A.ToDevice();
  B.ToDevice();
  C.ToDevice();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};

  cudaProfilerStart();
  for (int i = 0; i < kStreamNum; i++) {
    VecDot<<<block_num, threadnum, 0, streams[i]>>>(A.d, B.d, C.d, dim);
  }

  cudaDeviceSynchronize();
}

/*
 * 100 kernel works parallelly
 * another 100 kernel wait for several of them
 */
TEST(basic, huge_stream_num_with_event) {
  const int kStreamNum = 200;
  cudaStream_t streams[kStreamNum];
  cudaEvent_t events[kStreamNum / 2];  // only need half number of events
  for (int i = 0; i < kStreamNum; i++) {
    cudaStreamCreate(&streams[i]);
    if (i < kStreamNum / 2) {
      cudaEventCreate(&events[i], 0);
    }
  }

  // assign dependency
  int depend_ratio = 0.1;
  std::vector<std::vector<cudaEvent_t>> dependencies;
  for (int i = 0; i < kStreamNum / 2; i++) {
    dependencies.emplace_back();
    for (int j = 0; j < kStreamNum / 2; j++) {
      if (1. * rand() / RAND_MAX < depend_ratio) {
        dependencies.back().push_back(events[j]);
      }
    }
  }

  const int dim = 1000;
  HdMem<float> A(dim);
  HdMem<float> B(dim);
  HdMem<float> C(dim);

  RandVec<float>(A.h, dim);
  RandVec<float>(B.h, dim);
  RandVec<float>(C.h, dim);

  A.ToDevice();
  B.ToDevice();
  C.ToDevice();

  dim3 threadnum{256};
  dim3 block_num{(dim + 1) / threadnum.x};

  cudaProfilerStart();

  // launch the previous 100 kernels with unique stream each, and they work
  // concurrently.
  for (int i = 0; i < kStreamNum / 2; i++) {
    VecDot<<<block_num, threadnum, 0, streams[i]>>>(A.d, B.d, C.d, dim);
    cudaEventRecord(events[i], streams[i]);
  }

  // launch the last left kernels, with random dependency.
  for (int i = kStreamNum / 2; i < kStreamNum; i++) {
    for (auto event : dependencies[i - kStreamNum / 2]) {
      cudaStreamWaitEvent(streams[i], event, 0);
    }
    VecDot<<<block_num, threadnum, 0, streams[i]>>>(A.d, B.d, C.d, dim);
  }

  cudaDeviceSynchronize();
}

TEST(math, matrix_add) {
  int nx = 100;
  int ny = 200;

  HdMem<int> mem0(nx * ny);
  HdMem<int> mem1(nx * ny);
  HdMem<int> mem2(nx * ny);

  for (int i = 0; i < nx * ny; i++) {
    mem0.h[i] = i;
    mem1.h[i] = i;
  }
  mem0.ToDevice();
  mem1.ToDevice();

  int dx = 3;
  int dy = 3;

  dim3 block(dx, dy);
  dim3 grid((nx + dx - 1) / dx, (ny + dy - 1) / dy);

  SumMatrix<<<grid, block>>>(mem0.d, mem1.d, mem2.d, nx, ny);

  mem2.ToHost();

  cudaDeviceSynchronize();

  std::stringstream ss;
  for (int i = 0; i < 100; i++) {
    ss << mem2.h[i] << " ";
  }
  LOG(INFO) << "out " << ss.str();
}
