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
#include "paddle/fluid/framework/lite/target_wrapper.h"

namespace paddle {
namespace framework {
namespace lite {
namespace x86 {

template <typename Target>
class TargetWrapper {
 public:
  using stream_t = typename Target::stream_t;
  using event_t = Event<Target>;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size) { return new uint8_t[size]; }
  static void Free(void* ptr) { delete[] static_cast<uint8_t*>(ptr); }

  static void MemcpySync(void* dst, void* src, size_t size, Direction dir) {
    std::copy(static_cast<uint8_t*>(src), static_cast<uint8_t*>(src) + size,
              static_cast<uint8_t*>(dst));
  }
  static void MemcpyAsync(void* dst, void* src, size_t size,
                          const stream_t& stream, Direction dir) {
    MemcpySync(dst, src, size, dir);
  }
};
}  // namespace x86
}  // namespace framework
}  // namespace framework
}  // namespace paddle
