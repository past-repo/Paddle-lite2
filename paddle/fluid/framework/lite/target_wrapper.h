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
#include <iostream>

namespace paddle {
namespace framework {
namespace lite {

enum class TargetType { kHost = 0, kX86, kCUDA, kARM };
template <TargetType target>
struct Target {};

using Host = Target<TargetType::kHost>;
using CUDA = Target<TargetType::kCUDA>;
using X86 = Target<TargetType::kX86>;
using ARM = Target<TargetType::ARM>;

// Event sync for multi-stream devices like CUDA and OpenCL.
template <typename Target>
class Event {};

// Memory copy directions.
enum class Direction {
  HtoH = 0,
  HtoD,
  DtoH,
};

// This interface should be specified by each kind of target.
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

  static void* Malloc(size_t size) { return nullptr; }
  static void Free(void* ptr) {}

  static void MemcpySync(void* dst, void* src, size_t size, Direction dir) {}
  static void MemcpyAsync(void* dst, void* src, size_t size,
                          const stream_t& stream, Direction dir) {}
};

}  // namespace lite
}  // namespace framework
}  // namespace paddle
