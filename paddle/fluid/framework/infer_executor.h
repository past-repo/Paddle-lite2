#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/executor_utils.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

/*
 * Executor used for inference. Different from the original `Executor`, the
 * inference phase is much simpler than traning, so the executor should be much
 * clear and focus on performance issue for forward, such as data reusing and so
 * on.
 *
 * This executor's logic is quite clear:
 *  1. create operators
 *  2. create a scope
 *  3. execute the operators
 *
 */
class InferExecutor {
 public:
  explicit InferExecutor(const ProgramDesc& prog, int block_id,
                         const platform::Place& place)
      : place_(place) {
    CreateOps(prog, block_id);
  }

  // Init and check the feed and fetch support.
  // Naive implementation, just check all the feed and fetch matches.
  // TODO(Superjomn) improve this with IR latter.
  void InitForInference(const std::vector<std::string>& feeds,
                        const std::vector<std::string>& fetches) {
    LOG(WARNING) << "Inference mode activated";
    FLAGS_infer_mode = true;

    // Prepare or check fetches.
  }

  // This is for forward.
  void Run(Scope* scope) {
    PADDLE_ENFORCE(scope);
    for (auto& op : ops_) {
      op->Run(*scope, place_);
    }

    platform::DeviceContextPool::Instance().Get(place_)->Wait();
  }

  void Run(const ProgramDesc& prog, int block_id, Scope* scope,
           const std::map<std::string, const LoDTensor*>& feed_targets,
           std::map<std::string, const LoDTensor*>* fetch_targets,
           const std::string& feed_holder_name = "feed",
           const std::string& fetch_holder_name = "fetch") {
    auto& block = prog.Block(block_id);

    PADDLE_ENFORCE(
        has_feed_operators(global_block, *feed_targets, feed_holder_name),
        "Program in ExecutorPrepareContext should has feed_ops.");
    PADDLE_ENFORCE(
        has_fetch_operators(global_block, *fetch_targets, fetch_holder_name),
        "Program in the prepared context should has fetch_ops.");

    // Feed data.
    FeedDatas(scope, feed_targets, feed_holder_name);

    Run(scope);

    // Fetch data.
    FetchDatas(scope, *fetch_targets, fetch_holder_name);
  }

  void CreateVariables(const ProgramDesc& prog, Scope* scope, int block_id) {
    auto& block = prog.Block(block_id);
    const Scope* root_scope = scope;
    while (root_scope->parent()) {
      root_scope = root_scope->parent();
    }

    for (auto& var : block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) continue;
      // For parameters.
      if (var->Persistable()) {
        auto* ptr = const_cast<Scope*>(root_scope)->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create persiable variable " << var->Name()
                << " global, which pointer is " << ptr;
      } else {  // For local variables.
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create variable " << var->Name()
                << " locally, which pointer is " << ptr;
      }
    }
  }

  ~InferExecutor() { LOG(WARNING) << "InferExecutor deleted"; }

 protected:
  void CreateOps(const ProgramDesc& prog, int block_id) {
    PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), prog.Size());
    for (auto& op_desc : prog.Block(block_id).AllOps()) {
      ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
    }
  }

  // Used in inference, to set or get data from fluid.
  void FeedDatas(Scope* scope,
                 const std::map<std::string, const LoDTensor*>& data_map,
                 const std::string& holder_name);
  void FetchDatas(Scope* scope, std::map<std::string, LoDTensor*>& data_map,
                  const std::string& holder_name);
  void FeedData(Scope* scope, const LoDTensor& tensor,
                const std::string& holder_name, int idx);
  void FetchData(Scope* scope, LoDTensor* tensor,
                 const std::string& holder_name, int idx);

 private:
  std::vector<::std::unique_ptr<OperatorBase>> ops_;
  const platform::Place place_;
  std::vector<OperatorBase*> feed_ops_;
  std::vector<OperatorBase*> fetch_ops_;
};

}  // namespace framework
}  // namespace paddle
