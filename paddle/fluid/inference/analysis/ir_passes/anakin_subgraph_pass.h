#pragma once

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class AnakinSubgraphPass : public framework::ir::FusePassBase {
 public:
  std::unique_ptr<framework::ir::Graph> ApplyImpl(
      std::unique_ptr<framework::ir::Graph> graph) const override;

 private:
  void CreateAnakinOp(framework::ir::Node *x,
                      framework::ir::Graph *graph) const;
  void CleanIntermediateOutputs(framework::ir::Node *node);
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
