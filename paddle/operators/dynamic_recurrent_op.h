/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_TESTING
#include "gtest/gtest.h"
#endif

#include "paddle/framework/executor.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor_array.h"
#include "paddle/framework/variable.h"

namespace paddle {
namespace operators {

/**
 * Memory of a RNN (same as the role of `Momory` in PaddlePaddle).
 *
 * Memory attributes cached by this op, dims will be infered from
 * boot memories in father scope. Other attributes are copied from Op's proto
 * attributes.
 */
struct StateAttr {
  // name of current state variable
  std::string var;
  // name of previous step's state variable
  std::string pre_var;
  // name of the variables to init this memory (same role of `boot_layer` in
  // PaddlePaddle), which is store in father's scope.
  std::string boot_var;
};

struct Argument {
  std::string step_unit;
  std::string step_scopes;
  std::vector<std::string> inlinks;
  std::vector<std::string> outlinks;
  std::vector<StateAttr> states;
};

struct ArgumentName {
  std::string step_net;
  std::string step_scopes;
  std::string inlinks;
  std::string outlinks;
  std::string states;          // the memory name
  std::string ex_states;       // the previous memory name
  std::string initial_states;  // the boot memory name
  std::string block;
};

class RNNAlgorithm {
 public:
  enum ComputeMode { kForward = 0, kBackward = 1 };
  static const std::array<ArgumentName, 2> kArgNames;
  using value_type = float;

  /*
   * Different `Run` method for forward and backward, `_` is just for template
   * specifialization.
   */
  template <ComputeMode _>
  void Run(const framework::Scope& scope, const framework::OperatorBase& op,
           const platform::DeviceContext& dev_ctx);
  /*
   * Split the inputs(LoDTensors) to segments for each time step.
   */
  void SplitInputs();

  /*
   * Create step-scopes to store temporary outputs in each time steps.
   */
  void CreateScopes();

  /*
   * Link TensorArray steps to the corresponding variables located in
   * step-scopes.
   */
  void WriteStepInputs();

  /*
   * Write output of each step to the corresponding TensorArray.
   */
  void WriteStepOutputs();

  /*
   * Initialize the states, each state will have a corresponding pre-state,
   * which share the memory with the state in the previous time state. The
   * pre-state in the first time step will be initialized with an zero tensor or
   * a tensor in parent scope if is provided.
   */
  void InitStates();

  /*
   * Create state variables for each time step.
   */
  void CreateState(const StateAttr& state, size_t step);

  /*
   * Link pre-state variable in current scope to the state variable in the
   * previous time step (scope) by reference.
   */
  void LinkState(const StateAttr& state, size_t step);

  /*
   * Link state@GRAD when backwards, this will add `pre_state@GRAD` to
   * `state@GRAD` in the previous scope.
   */
  void LinkGradState(const StateAttr& state, size_t step);

  /*
   * Link the pre-state of the first time step to the `boot-state` in parent's
   * scope.
   */
  void LinkInitialState(const StateAttr& state);

  /*
   * Copy the gradient from `pre-state` in the first step-scope to the
   * `boot-state` in parent's scope, this is only used in backward mode.
   */
  void ExportInitialStateGradient(const StateAttr& state);

  /*
   * Create parameter gradient variables in step scopes to avoid the backward
   * algorithm change the variable in parent scope directly.
   */
  void CreateLocalParameterGradients();

  /*
   * Accumulate the weight gradient in all the time steps and sum to the
   * gradient in parent scope.
   */
  void ExportWeightGradients();

  /*
   * Calculate time steps.
   */
  void RunSteps();

  /*
   * Create an executor and run one step.
   */
  void RunOneStep(const platform::DeviceContext& dev_ctx,
                  const framework::OperatorBase& op,
                  framework::Scope* step_scope) {
    framework::Executor executor(dev_ctx);
    auto* block = op.Attr<framework::BlockDescBind*>(kArgNames[mode_].step_net);
    auto* program = block->Program();
    executor.Run(*program, step_scope, block->ID(),
                 false /*create_local_scope*/);
  }

  /*
   * Concatenate outputs in each time step and generate a LoDTensor.
   */
  void ConcatOutputs();

  void SetComputeMode(ComputeMode mode) { mode_ = mode; }
  bool IsForward() const { return mode_ == ComputeMode::kForward; }
  bool IsBackward() const { return mode_ == ComputeMode::kBackward; }

  /*
   * set a step unit that is created according to a RecurrentOp's step unit.
   */
  void SetStepUnit(std::unique_ptr<framework::OperatorBase> step_unit) {
    PADDLE_ENFORCE_NOT_NULL(step_unit);
    step_unit_ = std::move(step_unit);
  }
  const framework::OperatorBase& GetStepUnit() const { return *step_unit_; }

  const framework::TensorArray& state(const std::string& name) const {
    auto it = states_.find(name);
    PADDLE_ENFORCE(it != states_.end());
    return it->second;
  }

  const framework::TensorArray& pre_state(const std::string& name) const {
    auto it = pre_states_.find(name);
    PADDLE_ENFORCE(it != pre_states_.end());
    return it->second;
  }

  const framework::TensorArray& step_input(const std::string& name) const {
    auto it = step_inputs_.find(name);
    PADDLE_ENFORCE(it != step_inputs_.end());
    return it->second;
  }
  const framework::TensorArray& step_output(const std::string& name) const {
    auto it = step_outputs_.find(name);
    PADDLE_ENFORCE(it != step_outputs_.end());
    return it->second;
  }
  const framework::LoDTensor& step_tensor(size_t step,
                                          const std::string& name) const {
    PADDLE_ENFORCE_LT(step, cache_.scopes->size());
    auto var = cache_.scopes->at(step)->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(var, "no varialbe called %s exists in step %d",
                            name, step);
    return var->Get<framework::LoDTensor>();
  }

 protected:
  struct ArgCache {
    framework::Scope const* scope;
    std::vector<framework::Scope*>* scopes;
    std::map<std::string, framework::Variable*> inputs;
    std::map<std::string, framework::Variable*> outputs;
    platform::DeviceContext const* dev_ctx;
    const framework::OperatorBase* op;

    size_t num_steps{0};

    void Init(const ArgumentName& name, const framework::OperatorBase& op,
              const framework::Scope& scope,
              platform::DeviceContext const* dev_ctx, Argument* arg,
              bool is_grad);

    framework::Scope& GetScope(size_t index) {
      PADDLE_ENFORCE_LT(index, num_steps);
      return *scopes->at(index);
    }

    framework::LoDTensor* GetTensor(const framework::Scope& scope,
                                    const std::string& name);

   private:
    void InitArgument(const ArgumentName& name,
                      const framework::OperatorBase& op, Argument* arg,
                      bool is_grad);
    void CacheScopes(const framework::Scope& scope, const Argument& arg);
    void CacheInlinks(const framework::Scope& scope,
                      const std::vector<std::string>& names);
    void CacheOutlinks(const framework::Scope& scope,
                       const std::vector<std::string>& names);
    framework::Variable* GetVariable(const framework::Scope& scope,
                                     const std::string& name);
  };

 private:
  std::unique_ptr<framework::OperatorBase> step_unit_;
  std::unique_ptr<framework::Executor> step_executor_;
  std::map<std::string, framework::TensorArray> states_;
  std::map<std::string, framework::TensorArray> pre_states_;
  std::map<std::string, framework::TensorArray> step_inputs_;
  std::map<std::string, framework::TensorArray> step_outputs_;
  std::map<std::string, std::vector<framework::DySeqMeta>> dy_seq_metas_;
  // name of parameter variables
  std::vector<std::string> parameters_;
  Argument arg_;
  ArgCache cache_;
  ComputeMode mode_{ComputeMode::kForward};

#ifdef PADDLE_WITH_TESTING
  // test forward
  friend class RNNAlgorithmTestHelper;
  FRIEND_TEST(RNNAlgorithmTestHelper, SplitInputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, CreateCache);
  FRIEND_TEST(RNNAlgorithmTestHelper, CreateScopes);
  FRIEND_TEST(RNNAlgorithmTestHelper, WriteStepInputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, WriteStepOutputs);
  FRIEND_TEST(RNNAlgorithmTestHelper, InitStates);
  FRIEND_TEST(RNNAlgorithmTestHelper, ConcatOutputs);
// TODO(superjom) test backward
#endif
};

class DynamicRecurrentOp : public framework::OperatorBase {
 public:
  DynamicRecurrentOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  DynamicRecurrentOp(const DynamicRecurrentOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

  mutable RNNAlgorithm rnn;
};

class DynamicRecurrentGradientOp : public framework::OperatorBase {
 public:
  DynamicRecurrentGradientOp(const std::string& type,
                             const framework::VariableNameMap& inputs,
                             const framework::VariableNameMap& outputs,
                             const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  DynamicRecurrentGradientOp(const DynamicRecurrentGradientOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    PADDLE_THROW("Not implemented");
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

  mutable RNNAlgorithm rnn;
};

}  // namespace operators
}  // namespace paddle
