// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <glog/logging.h>
#include <thread>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/timer.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace inference {
namespace ut_helper {

using batch_updater_t = std::function<int(std::vector<PaddleTensor>*, int idx)>;

void MultiThreadRun(
    const std::vector<std::unique_ptr<PaddlePredictor>>& predictors,
    std::vector<std::vector<PaddleTensor>>* inputs,
    batch_updater_t&& batch_data_updater) {
  LOG(INFO) << "Run with " << predictors.size() << " threads";

  std::vector<std::thread> threads;

  for (size_t idx = 0; idx < predictors.size(); idx++) {
    auto& predictor = predictors[idx];
    threads.emplace_back([&, idx] {
      auto& input_slots = inputs->at(idx);
      std::vector<PaddleTensor> output_slots;

      double total_time = 0;
      int total_samples = 0;
      inference::Timer timer;
      int num_sample = 0;
      while ((num_sample = batch_data_updater(&input_slots, idx))) {
        timer.tic();
        CHECK(predictor->Run(input_slots, &output_slots));
        total_time += timer.toc();
        total_samples += num_sample;
      }

      float average_time = total_time / total_samples;
      LOG(INFO) << string::Sprintf("thread #%d average time per sample: %f",
                                   idx, average_time);
    });
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

}  // namespace ut_helper
}  // namespace inference
}  // namespace paddle
