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
#include <unordered_set>
#include "paddle/fluid/framework/lite/target_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
namespace paddle {
namespace framework {
namespace lite {

template <typename Target>
class EventTree {
 public:
  using event_t = Event<Target>;

  void AddChild(const event_t& event) { children_.push_back(event); }

  void Sync() {
    for (auto& event : children_) {
      TargetWrapper<Target>::SyncEvent(event);
    }
  }

 private:
  std::vector<event_t> children_;
};

// A wrapper on the original Tensor or LoDTensor.
template <typename Target>
class TensorWrapper {
 public:
  using event_tree_t = EventTree<Target>;

  void SyncEventTree() { event_tree_.Sync(); }
  void SetEventTree(event_tree_t&& tree) { event_tree_ = std::move(tree); }

 private:
  LoDTensor* tensor_;
  EventTree<Target> event_tree_;
};

}  // namespace framework
}  // namespace framework
}  // namespace paddle
