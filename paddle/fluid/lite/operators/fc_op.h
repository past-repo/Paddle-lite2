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

#include <string>
#include <vector>
#include "paddle/fluid/lite/op_lite.h"
#include "paddle/fluid/lite/tensor.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace framework {
namespace lite {
namespace operators {

struct FcParam {
  Tensor* input{nullptr};
  Tensor* w{nullptr};
  Tensor* bias{nullptr};
  Tensor* output{nullptr};
  // the input matrix dimentions.
  lite::DDim in_mat_dims;
  int in_num_col_dims{0};
};

class FcOpLite : public OpLite {
 public:
  FcOpLite() {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.input);
    CHECK_OR_FALSE(param_.output);
    CHECK_OR_FALSE(param_.w);
    // bias is optional.

    const auto input_dims = param_.input->dims();
    const auto w_dims = param_.w->dims();

    if (param_.bias) {
      const auto bias_dims = param_.bias->dims();
      if (bias_dims.size() == 2) {
        CHECK_EQ_OR_FALSE(bias_dims[0], 1);
        CHECK_EQ_OR_FALSE(bias_dims[1], w_dims[1]);
      } else if (bias_dims.size() == 1) {
        CHECK_EQ_OR_FALSE(bias_dims[0], w_dims[1]);
      }
    }

    CHECK_EQ_OR_FALSE(w_dims.size(), 2UL);
    CHECK_GT_OR_FALSE(input_dims.size(),
                      static_cast<size_t>(param_.in_num_col_dims));

    param_.in_mat_dims =
        lite::flatten_to_2d(input_dims, param_.in_num_col_dims);
    CHECK_EQ_OR_FALSE(param_.in_mat_dims[1], w_dims[0]);

    return true;
  }

  bool InferShape() const override {
    const auto input_dims = param_.input->dims();
    const auto w_dims = param_.w->dims();

    // Set output dims
    std::vector<int> output_dims(param_.in_num_col_dims + 1, 0);
    for (int i = 0; i < param_.in_num_col_dims; ++i) {
      output_dims[i] = input_dims[i];
    }
    output_dims.back() = w_dims[1];
    param_.output->Resize(output_dims);

    // share LoD
    // param_.output->set_lod(param_.input->lod());
    return true;
  }

  bool Run() override { return false; }

  bool Build(const framework::OpDesc& opdesc,
             framework::Scope* scope) override {
    return false;
  }

  std::string DebugString() const override { return "fc"; }

  void StaticPickKernel(const std::vector<OpTarget>& valid_targets) override {}

 private:
  mutable FcParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace framework
}  // namespace paddle
