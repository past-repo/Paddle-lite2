#include "paddle/fluid/inference/analysis/ir_passes/anakin_subgraph_pass.h"
#include "anakin_subgraph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

std::unique_ptr<framework::ir::Graph> AnakinSubgraphPass::ApplyImpl(
    std::unique_ptr<framework::ir::Graph> graph) const {
  return Pass::ApplyImpl(graph);
}

void AnakinSubgraphPass::CreateAnakinOp(framework::ir::Node *x,
                                        framework::ir::Graph *graph) const {
  auto *op_desc = node->Op();
}

void AnakinSubgraphPass::CleanIntermediateOutputs(framework::ir::Node *node) {}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
