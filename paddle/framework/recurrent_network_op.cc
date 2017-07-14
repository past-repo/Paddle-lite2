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

#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"
// #include "paddle/framework/op_registry.h"

#include <glog/logging.h>
#include <cstring>

namespace paddle {
namespace framework {

void RecurrentOp::Run(OpContext* contex) const {
  auto scope = contex->scope;

  PADDLE_ENFORCE(scope->HasVariable(net_name_), "step net is not in scope.");
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  LOG(INFO) << "create scopes";
  CreateScopes(scope);
  LOG(INFO) << "segment input";
  SegmentInputs(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get step scopes");
  // forward
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[0];
  LOG(INFO) << "sequence length " << seq_len;
  auto& scopes = *step_scopes->GetMutable<std::vector<ScopePtr>>();
  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    LOG(INFO) << "run step " << step_id;
    ScopePtr step_scope = scopes[step_id];
    // TODO replace memorys' copy with reference
    LinkMemories(scope, scopes, step_id);

    net->GetMutable<PlainNet>()->Run(step_scope);
  }

  LOG(INFO) << "concat outputs";
  // prepare outputs
  ConcatOutputs(scope);
}

void RecurrentOp::Init(const OpDesc& op_desc, AttributeMap& attrs) {
  OperatorBase::Init(op_desc, attrs);

  // set original inputs
  for (const std::string& input : op_desc.inputs()) {
    LOG(INFO) << "set input " << input;
    inputs_.push_back(input);
  }
  // set original outputs
  for (const std::string& output : op_desc.outputs()) {
    LOG(INFO) << "set output " << output;
    outputs_.push_back(output);
  }

  name_ = op_desc.name();
  net_name_ = inputs_.at(GetAttr<int>("step_net"));
  step_scopes_name_ = outputs_.back();

  // prepare inlinks
  PADDLE_ENFORCE(inlinks_.empty(), "RecurrentOp duplicate inited");
  LOG(INFO) << "set inlinks";
  for (auto id : GetAttr<std::vector<int>>("in_links")) {
    inlinks_.push_back(id);
  }
  PADDLE_ENFORCE(
      outputs_.size() > 1,
      "more than 1 output should be provided and the last is `step_scopes`");
  outlinks_ = std::vector<std::string>{outputs_.begin(), outputs_.end() - 1};

  // set memories
  auto memories = GetAttr<std::vector<std::string>>("memories");
  auto pre_memories = GetAttr<std::vector<std::string>>("pre_memories");
  PADDLE_ENFORCE(memories.size() == pre_memories.size(),
                 "The size of memories and pre_memories doesn't match: %d,%d.",
                 memories.size(), pre_memories.size());

  std::vector<std::string> boot_memories;
  LOG(INFO) << "set boot_memories";
  for (auto id : GetAttr<std::vector<int>>("boot_memories")) {
    boot_memories.push_back(inputs_[id]);
  }
  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories and boot_memories doesn't match: %d,%d",
                 memories.size(), boot_memories.size());
  for (size_t i = 0; i < memories.size(); ++i) {
    MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    memory_attrs_.push_back(mem_attr);
    LOG(INFO) << "set memorys:\t"
              << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }
}

void RecurrentOp::CreateScopes(ScopePtr scope) const {
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[0];
  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto step_scopes = scopes_var->GetMutable<std::vector<ScopePtr>>();
  // TODO Only two scopes are needed for inference, this case will be
  // supported later.
  if (seq_len > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len; ++i) {
      step_scopes->push_back(std::make_shared<Scope>(scope));
    }
  }
}

void RecurrentOp::SegmentInputs(ScopePtr scope) const {
  PADDLE_ENFORCE(!inlinks_.empty(), "no in links are provided.");
  auto inlink_alias = GetAttr<std::vector<std::string>>("in_link_alias");
  PADDLE_ENFORCE(inlinks_.size() == inlink_alias.size(),
                 "in_links/in_link_alias mismatch.");

  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto& step_scopes = *scopes_var->GetMutable<std::vector<ScopePtr>>();
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  int seq_len = dims[0];
  int batch_size = dims[1];
  for (size_t i = 0; i < inlinks_.size(); ++i) {
    auto input_dims = Input(scope, inlinks_[i])->GetMutable<Tensor>()->dims();
    int input_dim = input_dims[2];
    int length = batch_size * input_dim;
    const float* scope_input =
        Input(scope, inlinks_[i])->GetMutable<Tensor>()->data<float>();
    for (int j = 0; j < seq_len; j++) {
      Variable* input_var = step_scopes[j]->CreateVariable(inlink_alias[i]);
      Tensor* step_input_tensor = input_var->GetMutable<Tensor>();
      float* step_input = step_input_tensor->mutable_data<float>(
          make_ddim({batch_size, input_dim}), platform::CPUPlace());
      std::memcpy(step_input, scope_input + j * length, length);
    }
  }
}

void RecurrentOp::ConcatOutputs(ScopePtr scope) const {
  auto outlink_alias = GetAttr<std::vector<std::string>>("out_link_alias");
  PADDLE_ENFORCE(outlinks_.size() == outlink_alias.size(),
                 "out_links/out_link_alias mismatch.");

  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto& step_scopes = *scopes_var->GetMutable<std::vector<ScopePtr>>();
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  int seq_len = dims[0];
  int batch_size = dims[1];
  for (size_t i = 0; i < outlinks_.size(); i++) {
    auto output_dims = step_scopes[0]
                           ->GetVariable(outlink_alias[0])
                           ->GetMutable<Tensor>()
                           ->dims();
    int output_dim = output_dims[1];
    int length = batch_size * output_dim;
    Tensor* output_tensor =
        scope->CreateVariable(outlinks_[i])->GetMutable<Tensor>();
    float* output = output_tensor->mutable_data<float>(
        make_ddim({seq_len, batch_size, output_dim}), platform::CPUPlace());
    for (int j = 0; j < seq_len; j++) {
      Variable* output_var = step_scopes[j]->GetVariable(outlink_alias[i]);
      const float* step_output =
          output_var->GetMutable<Tensor>()->data<float>();
      std::memcpy(output + j * length, step_output, length);
    }
  }
}

void RecurrentOp::LinkMemories(ScopePtr scope,
                               std::vector<ScopePtr>& step_scopes,
                               size_t step) const {
  PADDLE_ENFORCE(step < step_scopes.size(),
                 "step [%d] out of range of step scopes' size [%d]", step,
                 step_scopes.size());
  auto step_scope = step_scopes[step];
  // copy boot memory
  for (auto& attr : memory_attrs_) {
    Tensor* boot_tensor{nullptr};
    if (step == 0) {
      PADDLE_ENFORCE(scope->HasVariable(attr.boot_var),
                     "memory [%s]'s boot variable [%s] not exists", attr.var,
                     attr.boot_var);
      // update memory's ddim
      boot_tensor = scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
      attr.dims = boot_tensor->dims();
    }
    Variable* memory_var = step_scope->CreateVariable(attr.pre_var);

    // TODO the memory of current step should be allocaled in step net ?
    Tensor* cur_memory =
        step_scopes[step]->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_memory->mutable_data<float>(attr.dims, platform::CPUPlace());

    // copy from boot memory
    // TODO support more device
    // TODO mutable_data is currently invalid
    float* memory_tensor_val =
        memory_var->GetMutable<Tensor>()->mutable_data<float>(
            attr.dims, platform::CPUPlace());
    if (step == 0) {
      PADDLE_ENFORCE(boot_tensor, "boot_tensor should be retrieved before");
      // copy from boot memory
      std::memcpy(memory_tensor_val, boot_tensor->data<float>(),
                  product(attr.dims));
    } else {
      // copy from previous step scope's memory to this scope's
      // `pre - memory`
      Tensor* pre_step_memory =
          step_scopes[step - 1]->GetVariable(attr.var)->GetMutable<Tensor>();

      std::memcpy(memory_tensor_val, pre_step_memory->data<float>(),
                  product(attr.dims));
    }
  }
}

// TODO testing when including operator.h

// class RecurrentOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
//  public:
//   RecurrentOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
//       : OpProtoAndCheckerMaker(proto, op_checker) {
//     // AddInput("input", "input of test op"); // need to support dynamic
//     number
//     // AddOutput("output", "output of test op"); // need to support dynamic
//     number
//     AddAttr<std::std::vector<int>>("in_links", "The input link positions in
//     the all inputs.")
//         .SetDefault({0});
//     AddAttr<std::std::vector<int>>("boot_memories", "The initial memory
//     positions in the all inputs.");
//     AddAttr<int>("step_net", "The step net position in the all inputs.");
//
//     AddAttr<std::std::vector<std::string>>("in_link_alias", "The input link
//     alias in the step network.");
//     AddAttr<std::std::vector<std::string>>("out_link_alias", "The output link
//     alias in the step network.");
//     AddAttr<std::std::vector<std::string>>("memories", "The memory names.");
//     AddAttr<std::std::vector<std::string>>("pre_memories", "The
//     history/previous memory names.");
//
//     AddType("recurrent_op");
//     AddComment("This is a recurrent group operator.");
//   }
// };
//
// REGISTER_OP(recurrent_op, RecurrentOp, RecurrentOpProtoAndCheckerMaker);

}  // namespace framework
}  // namespace paddle
