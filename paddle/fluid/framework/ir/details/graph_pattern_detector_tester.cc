#include "paddle/fluid/framework/ir/details/graph_pattern_detector.h"
#include <gtest/gtest.h>
#include "../graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/ut_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace details {

void BuildPattern(PDPattern* pattern, PDNode** start) {
  // mark o2, o3, v2

  // The pattern is a graph:
  //   o2(a node named o2) -> v2(a node named v2)
  //   v2 -> o3(a node named o3)
  auto* o2 = pattern->NewNode([](Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->Name() == "op2" && node->IsOp();
  });
  auto* o3 = pattern
                 ->NewNode([](Node* node) {
                   // The teller can be any condition, such as op type, or
                   // variable's shape.
                   return node && node->Name() == "op3";
                 })
                 ->assert_more([](Node* node) { return node->IsOp(); });

  auto* v2 = pattern->NewNode([](Node* node) {
    // The teller can be any condition, such as op type, or variable's shape.
    return node && node->Name() == "var2" && node->IsVar();
  });

  ASSERT_FALSE(o2->Tell(nullptr));
  ASSERT_FALSE(o3->Tell(nullptr));
  ASSERT_FALSE(v2->Tell(nullptr));

  pattern->AddEdge(o2, v2);
  pattern->AddEdge(v2, o3);

  ASSERT_EQ(pattern->edges().size(), 2UL);
  ASSERT_EQ(pattern->edges()[0].first, o2);
  ASSERT_EQ(pattern->edges()[0].second, v2);
  ASSERT_EQ(pattern->edges()[1].first, v2);
  ASSERT_EQ(pattern->edges()[1].second, o3);

  *start = o3;
}

TEST(GraphPatternDetector, CollectAllNodeRoles) {
  ProgramDesc program;
  Graph graph(program);
  BuildGraph(&graph);

  PDNode* thestart;
  PDPattern pattern;
  BuildPattern(&pattern, &thestart);

  std::unordered_map<Node*, std::unordered_set<PDNode*>> roles;
  CollectAllNodeRoles(graph, pattern, &roles);

  LOG(INFO) << "roles " << roles.size();

  for (auto& item : roles) {
    LOG(INFO) << "has " << item.second.size() << " roles";
  }
}

TEST(GraphPatternDetector, GetStartOfPattern) {
  ProgramDesc program;
  Graph graph(program);
  BuildGraph(&graph);

  PDPattern pattern;
  PDNode* thestart;
  BuildPattern(&pattern, &thestart);

  PDNode* start{nullptr};
  GetStartOfPattern(pattern, &start);

  LOG(INFO) << "start " << start;
  ASSERT_EQ(thestart, start);
}

TEST(GraphPatternDetector, TraversePatternGraphUndirecttly) {

}

}  // namespace details
}  // namespace ir
}  // namespace framework
}  // namespace paddle
