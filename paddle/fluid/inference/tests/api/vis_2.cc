#include <gtest/gtest.h>
#include <iostream>
#include <mutex>
#include <thread>
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "time.h"

using namespace paddle;

TEST(vis, multi_thread) {
  contrib::AnalysisConfig config(true);
  config.prog_file =
      "/home/chunwei/project2/models/detect.models/product_common_det300/"
      "__model__";
  config.param_file =
      "/home/chunwei/project2/models/detect.models/product_common_det300/"
      "params";
  config.fraction_of_gpu_memory = 0.4;

  auto predictor = CreatePaddlePredictor(config);
  std::vector<std::thread> threads;
  std::vector<size_t> latencies;
  inference::Timer timer;

  std::vector<std::vector<PaddleTensor>> input_datas;

  for (int i = 0; i < 1000; i++) {
    input_datas.emplace_back();
    auto& inputs = input_datas.back();
    inputs.resize(2);

    auto& tensor0 = inputs[0];
    auto& tensor1 = inputs[1];

    tensor0.shape.assign({1, 3, 225, 300});
    tensor0.data.Resize(1 * 3 * 225 * 300 * sizeof(float));

    for (int i = 0; i < tensor0.data.length() / sizeof(float); i++) {
      static_cast<float*>(tensor0.data.data())[i] =
          rand() / RAND_MAX * 256 - 127;
    }

    tensor1.shape.assign({3, 1});
    tensor1.data.Resize(3 * 1 * sizeof(float));
    auto* tensor1_data = static_cast<float*>(tensor1.data.data());
    tensor1_data[0] = 2.25e2;
    tensor1_data[1] = 3e2;
    tensor1_data[2] = 2.9e-1;
  }

  std::mutex mut;
  for (int i = 0; i < 3; i++) {
    threads.emplace_back([&] {
      auto child = predictor->Clone();
      // auto child = CreatePaddlePredictor(config);
      // auto child = CreatePaddlePredictor(config);
      for (auto& inputs : input_datas) {
        std::vector<PaddleTensor> outputs;
        timer.tic();
        ASSERT_TRUE(child->Run(inputs, &outputs));
        auto time = timer.toc();
        std::lock_guard<std::mutex> lk(mut);
        latencies.push_back(time);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  LOG(INFO) << "latencies:";
  for (auto v : latencies) {
    std::cout << v << std::endl;
  }
}
