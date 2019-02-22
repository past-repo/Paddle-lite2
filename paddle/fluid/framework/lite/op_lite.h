// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <boost/variant.hpp>
#include <map>
#include <string>
#include "paddle/fluid/framework/lite/context.h"
#include "paddle/fluid/framework/lite/op_kernel.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
namespace lite {

using any_t = boost::variant<int, float, framework::Variable *>;
using anys_t = std::map<std::string, any_t>;

/**
 * The base class of an light-weight operators, currently just used in inference
 * to eliminate overhead of some operations in current framework.
 *
 * The Operator are designed as follows:
 * - it can has some members to hold the argument addresses,
 * - it should act just like a function call, no more logic should included.
 */
class OpLite {
 public:
  enum class KernelStrategy {
    // Return the user specified one.
    kStatic = 0,
    // Specify the expected kernel externally.
    kSpecified,
    // Run each kernel to evaluate and get the best kernel.
    kRuntime,
  };

  OpLite(OpContext &&x) : op_context_(std::move(x)) {}

  virtual bool CheckShape() const { return true; }
  virtual bool InferShape() const { return true; }
  virtual bool Run() = 0;
  virtual bool Build(const framework::OpDesc &opdesc,
                     framework::Scope *scope) = 0;
  virtual std::string DebugString() const = 0;

  virtual std::string StaticPickKernel(
      const std::vector<OpTarget> &valid_targets) = 0;

  static std::unique_ptr<any_kernel_t> PickBestKernel(
      const std::string &op_type, const std::vector<OpPlace> &valid_places,
      KernelStrategy kerlel_strategy = KernelStrategy::kStatic);

 private:
  virtual ~OpLite() = default;
  OpContext op_context_;
};

#define CHECK_OR_FALSE(cond)               \
  if (!(cond)) {                           \
    LOG(ERROR) << #cond << " test error!"; \
    return false;                          \
  }
#define CHECK_EQ_OR_FALSE(a__, b__)                           \
  if ((a__) != (b__)) {                                       \
    LOG(ERROR) << #a__ << " == " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " != " << b__;                       \
    return false;                                             \
  }

#define CHECK_GT_OR_FALSE(a__, b__)                          \
  if (!((a__) > (b__))) {                                    \
    LOG(ERROR) << #a__ << " > " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " <= " << b__;                      \
    return false;                                            \
  }

#define CHECK_GE_OR_FALSE(a__, b__)                           \
  if (!((a__) >= (b__))) {                                    \
    LOG(ERROR) << #a__ << " >= " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " < " << b__;                        \
    return false;                                             \
  }

}  // namespace lite
}  // namespace framework
}  // namespace paddle
