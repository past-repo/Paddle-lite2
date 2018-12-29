/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/dot.h"

namespace paddle {
namespace framework {
namespace ir {

const char kGraphvizMarkedNodeAttr[] = "__graphviz__marked_node__";

class GraphVizPass : public Pass {
 public:
  using marked_nodes_t = std::unordered_set<const Node *>;

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

  // Tell whether there are any marked nodes in the graph. Consume the
  // corresponding attribute.
  marked_nodes_t ConsumeMarkedNodes(const Graph *graph) const;
};

// Generate a DOT program to describe the graph.
// node_id_offset: to make the nodes' names in multiple graphs unique.
void DotDrawGraph(const Graph &graph, paddle::inference::analysis::Dot *dotter,
                  int node_id_offset = 0,
                  const std::unordered_set<const Node *> &marked_nodes =
                      std::unordered_set<const Node *>());

static GraphVizPass::marked_nodes_t &GetMarkedNodes(Graph *graph) {
  if (!graph->Has(kGraphvizMarkedNodeAttr)) {
    graph->Set(kGraphvizMarkedNodeAttr, new GraphVizPass::marked_nodes_t);
  }
  return graph->Get<GraphVizPass::marked_nodes_t>(kGraphvizMarkedNodeAttr);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
