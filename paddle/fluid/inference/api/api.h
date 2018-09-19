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

#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/inference/api/public/paddle_inference_api.h"

namespace paddle {

// Configurations for Anakin engine.
struct AnakinConfig : public PaddlePredictor::Config {
  enum TargetType { NVGPU = 0, X86 };
  int device;
  std::string model_file;
  int max_batch_size{-1};
  TargetType target_type;
};

struct TensorRTConfig : public NativeConfig {
  // Determine whether a subgraph will be executed by TRT.
  int min_subgraph_size{1};
  // While TensorRT allows an engine optimized for a given max batch size
  // to run at any smaller size, the performance for those smaller
  // sizes may not be as well-optimized. Therefore, Max batch is best
  // equivalent to the runtime batch size.
  int max_batch_size{1};
  // For workspace_size, refer it from here:
  // https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#troubleshooting
  int workspace_size{1 << 30};
};

// NOTE WIP, not stable yet.
struct AnalysisConfig : public NativeConfig {
  //
  enum class IrPassMode {
    kSystem,   // Use system default passes, not customize.
    kInclude,  // Specify the passes in `ir_passes`.
    kExclude   // Specify the disabled passes in `ir_passes`.
  };

  bool enable_ir_optim = true;
  IrPassMode ir_mode{IrPassMode::kExclude};
  // attention lstm fuse works only on some specific models, disable as default.
  std::vector<std::string> ir_passes{"attention_lstm_fuse_pass"};
};

}  // namespace paddle
