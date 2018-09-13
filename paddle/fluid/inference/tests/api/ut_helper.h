#pragma once
#include <glog/logging.h>
#include <thread>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/timer.h"

namespace paddle {
namespace inference {
namespace ut_helper {

using batch_updater_t = std::function<int(std::vector<PaddleTensor>*, int idx)>;

void MultiThreadRun(const std::vector<std::unique_ptr<PaddlePredictor>> &predictors,
                    std::vector<std::vector<PaddleTensor>> *inputs,
                    batch_updater_t &&batch_data_updater) {
  LOG(INFO) << "Run with " << predictors.size() << " threads";

  std::vector<std::thread> threads;

  int idx = 0;
  for (auto& predictor : predictors) {
    threads.emplace_back([&] {
      auto& input_slots = inputs->at(idx);
      std::vector<PaddleTensor> output_slots;

      double total_time = 0;
      inference::Timer timer;
      while (batch_data_updater(&input_slots, idx++)) {
        timer.tic();
        CHECK(predictor->Run(input_slots, &output_slots));
        total_time += timer.toc();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace ut_helper
}  // namespace inference
}  // namespace paddle
