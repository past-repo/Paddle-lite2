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

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {

using framework::ir::kParamScopeAttr;

bool AnalysisPredictor::Init(
    const std::shared_ptr<framework::Scope>& parent_scope,
    framework::ProgramDesc* program = nullptr) {
  VLOG(3) << "Predictor::init()";
#if !defined(_WIN32)
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";
    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }
#endif

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
    LOG(WARNING) << "ir optimize only supports CPU currently";
    config_.enable_ir_optim = false;
  } else {
    place_ = paddle::platform::CPUPlace();
  }

  // Set scope.
  if (parent_scope) {
    LOG(INFO) << "Inherit parent scope from " << parent_scope;
    scope_ = parent_scope;
    scope_inherited_ = true;
  } else {
    LOG(INFO) << "Create new parent scope";
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
    scope_inherited_ = false;
  }

  executor_.reset(new paddle::framework::Executor(place_));

  if (!program) {
    // Initialize the inference program
    if (!config_.model_dir.empty()) {
      // Parameters are saved in separate files sited in
      // the specified `dirname`.
      inference_program_ = paddle::inference::Load(
          executor_.get(), scope_.get(), config_.model_dir);
    } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
      // All parameters are saved in a single file.
      // The file names should be consistent with that used
      // in Python API `fluid.io.save_inference_model`.
      inference_program_ = paddle::inference::Load(
          executor_.get(), scope_.get(), config_.prog_file, config_.param_file);
    } else {
      LOG(ERROR) << "fail to load inference model.";
      return false;
    }

    OptimizeInferenceProgram();
  } else {
    inference_program_.reset(program);
  }

  sub_scope_ = &scope_->NewScope();

  ctx_ = executor_->Prepare(*inference_program_, 0);
  if (config_._use_mkldnn) {
    executor_->EnableMKLDNN(*inference_program_);
  }

  VLOG(5) << "to create variables";
  executor_->CreateVariables(*inference_program_, sub_scope_, 0);
  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

void AnalysisPredictor::OptimizeInferenceProgram() {
  LOG(INFO) << "optimize begin";
  FLAGS_IA_enable_ir = config_.enable_ir_optim;
  FLAGS_IA_enable_tensorrt_subgraph_engine = false;
  FLAGS_IA_output_storage_path = "";  // Don't output the model.

  // prepare inference_program
  if (!config_.model_dir.empty()) {
    argument_.fluid_model_dir.reset(new std::string(config_.model_dir));
  } else {
    PADDLE_ENFORCE(
        !config_.param_file.empty(),
        "Either model_dir or (param_file, prog_file) should be set.");
    PADDLE_ENFORCE(!config_.prog_file.empty());
    argument_.fluid_model_program_path.reset(
        new std::string(config_.prog_file));
    argument_.fluid_model_param_path.reset(new std::string(config_.param_file));
  }
  argument_.origin_program_desc.reset(
      new ProgramDesc(*inference_program_->Proto()));

  // Set scope for parameter modification.
  PADDLE_ENFORCE(
      !scope_inherited_,
      "the parameters in an inheriented scope can not be optimized.");
  argument_.Set(framework::ir::kParamScopeAttr, scope_.get());

  // Run ir passes.
  PADDLE_ENFORCE(config_.ir_mode == AnalysisConfig::IrPassMode::kExclude,
                 "Only kExclude is supported yet.");
  Analyzer().DisableIrPasses(config_.ir_passes).Run(&argument_);

  CHECK(argument_.transformed_program_desc);
  VLOG(5) << "to prepare executor";
  inference_program_.reset(
      new framework::ProgramDesc(*argument_.transformed_program_desc));

  // Update scope.
  if (argument_.Has(kParamScopeAttr)) {
    if (&argument_.Get<framework::Scope>(kParamScopeAttr) != scope_.get()) {
      LOG(INFO) << "reset scope";
      scope_.reset(
          argument_.Release<framework::Scope>(framework::ir::kParamScopeAttr));
    } else {
      argument_.Release<framework::Scope>(framework::ir::kParamScopeAttr);
    }
  }
  LOG(INFO) << "== optimize end ==";
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    AnalysisConfig, PaddleEngineKind::kAnalysis>(const AnalysisConfig& config) {
  VLOG(3) << "create AnalysisConfig";
  if (config.use_gpu) {
    // 1. GPU memeroy
    PADDLE_ENFORCE_GT(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         std::to_string(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new AnalysisPredictor(config));
  if (!dynamic_cast<AnalysisPredictor*>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
  return predictor;
}

std::unique_ptr<PaddlePredictor> AnalysisPredictor::Clone() {
  VLOG(3) << "Predictor::clone";
  LOG(INFO) << "Cloning predictor";
  std::unique_ptr<PaddlePredictor> cls(new AnalysisPredictor(config_));

  auto* program = new framework::ProgramDesc(*inference_program_->Proto());
  if (!dynamic_cast<AnalysisPredictor*>(cls.get())->Init(scope_, program)) {
    LOG(ERROR) << "fail to call Init";
    return nullptr;
  }
#ifdef __clang__
  // fix clang compile error
  return cls;
#else
  // fix manylinux compile error.
  return std::move(cls);
#endif
}

}  // namespace paddle
