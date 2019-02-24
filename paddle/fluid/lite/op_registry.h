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

#include <memory>
#include <string>
#include <unordered_map>
#include "op_lite.h"
#include "target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace framework {
namespace lite {

using KernelFunc = std::function<void()>;
using KernelFuncCreator = std::function<std::unique_ptr<KernelFunc>()>;

template <typename ItemType>
class Factory {
 public:
  using item_t = ItemType;
  using self_t = Factory<item_t>;
  using item_ptr_t = std::unique_ptr<item_t>;
  using creator_t = std::function<item_ptr_t()>;

  static Factory& Global() {
    static Factory* x = new self_t;
    return *x;
  }

  void Register(const std::string& op_type, creator_t&& creator) {
    CHECK(!creators_.count(op_type)) << "The op " << op_type
                                     << " has already registered";
    creators_.emplace(op_type, std::move(creator));
  }

  item_ptr_t Create(const std::string& op_type) const {
    auto it = creators_.find(op_type);
    CHECK(it != creators_.end());
    return it->second();
  }

 protected:
  std::unordered_map<std::string, creator_t> creators_;
};

class LiteOpRegistry final : public Factory<OpLite> {
 public:
  static LiteOpRegistry& Global() {
    static auto* x = new LiteOpRegistry;
    return *x;
  }

 private:
  LiteOpRegistry() = default;
};

template <TargetType Target, PrecisionType Precision>
class KernelRegistryForTarget : public Factory<OpKernel<Target, Precision>> {};

class KernelRegistry final {
 public:
  KernelRegistry() {
#define INIT_FOR(target__, precision__)                                    \
  registries_[KernelRegistry::GetKernelOffset<TARGET(target__),            \
                                              PRECISION(precision__)>()] = \
      &KernelRegistryForTarget<TARGET(target__),                           \
                               PRECISION(precision__)>::Global();
    // Currently, just register 2 kernel targets.
    INIT_FOR(kARM, kFloat);
    INIT_FOR(kHost, kFloat);
#undef INIT_FOR
  }

  static KernelRegistry& Global() {
    static auto* x = new KernelRegistry;
    return *x;
  }

  template <TargetType Target, PrecisionType Precision>
  void Register(const std::string& name,
                typename KernelRegistryForTarget<Target, Precision>::creator_t&&
                    creator) {
    using kernel_registor_t = KernelRegistryForTarget<Target, Precision>;
    any_cast<kernel_registor_t*>(
        registries_[GetKernelOffset<Target, Precision>()])
        ->Register(name, std::move(creator));
  }

  // Get a kernel registry offset in all the registries.
  template <TargetType Target, PrecisionType Precision>
  static constexpr int GetKernelOffset() {
    return kNumTargets * static_cast<int>(Target) + static_cast<int>(Precision);
  }

 private:
  std::array<any, kNumTargets * kNumPrecisions> registries_;
};

}  // namespace lite
}  // namespace framework
}  // namespace paddle

#define LITE_OP_REGISTER(op_type__) op_type__##__registry__
#define LITE_OP_REGISTER_INSTANCE(op_type__) op_type__##__registry__instance__
#define REGISTER_LITE_OP(op_type__, OpClass)                         \
  struct LITE_OP_REGISTER(op_type__) {                               \
    LITE_OP_REGISTER(op_type__)() {                                  \
      paddle::framework::lite::LiteOpRegistry::Global().Register(    \
          #op_type__,                                                \
          []() -> std::unique_ptr<paddle::framework::lite::OpLite> { \
            return std::unique_ptr<paddle::framework::lite::OpLite>( \
                new OpClass);                                        \
          });                                                        \
    }                                                                \
  };                                                                 \
  static LITE_OP_REGISTER(op_type__) LITE_OP_REGISTER_INSTANCE(op_type__);

#define KERNEL_REGISTER(op_type__, type__, precision__) \
  op_type__##type__##precision__##__registor__
#define KERNEL_REGISTER_INSTANCE(op_type__, type__, precision__) \
  op_type__##type__##precision__##__registor__instance__

#define REGISTER_LITE_KERNEL(op_type__, target__, precision__, KernelClass) \
  struct KERNEL_REGISTER(op_type__, target__, precision_) {                 \
    KERNEL_REGISTER(op_type__, target__, precision_)() {                    \
      paddle::framework::lite::KernelRegistry::Global()                     \
          .Register<TARGET(target__), PRECISION(precision__)>(              \
              [] { return new KernelClass<target__, precision__>; });       \
    }                                                                       \
  };                                                                        \
  static KERNEL_REGISTER(op_type__, type__, precision__)                    \
      KERNEL_REGISTER_INSTANCE(op_type__, type__, precision__);
