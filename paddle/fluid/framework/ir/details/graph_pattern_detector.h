#include <queue>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace details {
/*
 * Match graph pattern using the similar method of substring match.
 */

// Node with roles of PDNode.
using node_roles_t = std::unordered_map<Node*, std::unordered_set<PDNode*>>;
using role_t = std::pair<Node*, PDNode*>;
// PDNode linking direction.
struct Direction {
  Direction(PDNode* n, PDNode* l, bool r)
      : node(n), linked_to(l), reversed(r) {}

  PDNode* node;
  PDNode* linked_to;
  bool reversed;
};
using direct_t = Direction;
using connected_map_t = std::unordered_map<PDNode*, std::vector<direct_t>>;

// Collect all the nodes' roles in an ir graph.
void CollectAllNodeRoles(const Graph& graph, const PDPattern& pattern,
                         node_roles_t* roles) {
  for (auto* node : graph.Nodes()) {
    for (auto& pdnode : pattern.nodes()) {
      if (pdnode->Tell(node)) {
        (*roles)[node].insert(pdnode.get());
      }
    }
  }
}

// Get the start point of a pattern, a start point match will trigger the
// pattern matching, to lower the frequency, the most specific PDNode will be
// selected as the start point.
void GetStartOfPattern(const PDPattern& pattern, PDNode** start) {
  PADDLE_ENFORCE_GT(pattern.nodes().size(), 0UL);
  size_t max_size = 0;
  for (auto& pdnode : pattern.nodes()) {
    if (pdnode->asserts().size() > max_size) {
      *start = pdnode.get();
      max_size = pdnode->asserts().size();
    }
  }
}

// Get the order of visiting nodes in the Pattern.
// @pattern: the pattern.
// @order: the order to visit the nodes in a PDPattern.
// @direction: mapping from one PDNode to another which is connectted with
//    direction.
void TraversePatternGraphUndirecttly(const PDPattern& pattern, PDNode* start,
                                     std::vector<direct_t>* order,
                                     connected_map_t* pat_connection) {
  PADDLE_ENFORCE_NOT_NULL(start);
  // Create pat_connection first.
  for (auto& edge : pattern.edges()) {
    (*pat_connection)[edge.first].emplace_back(edge.first, edge.second, false);
    (*pat_connection)[edge.second].emplace_back(edge.second, edge.first,
                                                true /*reverse*/);
  }

  // DFS
  std::unordered_set<PDNode*> visited;
  std::queue<direct_t> queue;
  while (!queue.empty()) {
    // visit cur
    auto top = queue.front();
    queue.pop();
    order->push_back(top);
    visited.insert(top.linked_to);

    // append nearby connected.
    PADDLE_ENFORCE(pat_connection->count(top.linked_to),
                   "Invalid graph pattern detected, the nodes in a PDPattern "
                   "should be fully connected to each other.");
    for (const auto& connected : pat_connection->at(top.linked_to)) {
      if (!visited.count(connected.node)) {
        queue.emplace(connected.linked_to, connected.node, !connected.reversed);
      }
    }
  }
  PADDLE_ENFORCE_EQ(pattern.nodes().size(), order->size(),
                    "Invalid graph pattern detected, the nodes in the pattern "
                    "should be fully-conncted.");
}

// // Check whether a node assign with a role given the existing role assignment.
// bool PartialRolesMatch(std::unordered_map<Node*, PDNode*>& existing_roles,
//                          const connected_map_t& pat_connection, Node* node,
//                          PDNode* role) {
//   // The node has been assigned role, can not be assigned with another role.
//   if (existing_roles.count(node)) return false;
//   std::unordered_map<PDNode*, Node*> rev_roles;
//   for (auto& role : existing_roles) {
//     rev_roles[role.second] = role.first;
//   }

//   // Check pat_connection
//   for (const auto& linked_to : pat_connection.at(role)) {
//     PADDLE_ENFORCE_EQ(linked_to.node, role);
//     auto linked_to_node = rev_roles.find(linked_to.linked_to);

//     // The connected node has no role, no need to check.
//     if (linked_to_node == rev_roles.end()) continue;  // NOLINT

//     // Check the connected nodes from the inputs and outputs.
//     if (linked_to.reversed &&
//         std::find(node->inputs.begin(), node->inputs.end(), *linked_to_node) ==
//             node->inputs.end()) {
//       return false;
//     }
//     if (!linked_to.reversed &&
//         std::find(node->outputs.begin(), node->outputs.end(),
//                   *linked_to_node) == node->outputs.end()) {
//       return false;
//     }
//   }
//   return true;
// }

// // Recursively match a pattern start from a node in the pattern.
// // The phase:
// // - Check the Node matches the PDPattern, check the edges linking to/from this
// // Node,
// // - Add this Node and the corresponding PDNode to the role map,
// // - Continue to match with the connected Node and PDNode.
// bool PatternMatch(const Graph& graph, const std::vector<direct_t>& pat_order,
//                   const connected_map_t& pat_connection, Node* node,
//                   int pat_order_index, const node_roles_t& node_roles,
//                   std::unordered_map<Node*, PDNode*>* role,
//                   bool link_reversed) {
//   PADDLE_ENFORCE(!graph.Nodes().empty());
//   PADDLE_ENFORCE(!pat_order.empty());
//   PADDLE_ENFORCE_NOT_NULL(node);
//   PADDLE_ENFORCE_GE(pat_order_index, 0);
//   PADDLE_ENFORCE(!node_roles.empty());
//   PADDLE_ENFORCE_NOT_NULL(role);
//   if (pat_order_index >= pat_order.size()) return true;

//   const auto& pat_dire = pat_order[pat_order_index];
//   // Check whether the current Node matches the PDNode.
//   if (!node_roles.at(node).count(pat_dire.node)) return false;

//   if (!PartialRolesMatch(*role, pat_connection, node, pat_dire.node))
//     return false;

//   // Assign a role.
//   (*role)[node] = pat_dire.node;

//   // Continue to match the existing connected nodes, DFS, return if any is
//   // matched, because the node in the graph can only assign with one role.
//   for (auto* x : node->inputs) {
//     if (!role->count(x)) {
//       if (PatternMatch(graph, pat_order, pat_connection, x, pat_order_index + 1,
//                        node_roles, role, true /*reversed*/))
//         return true;
//     }
//   }
//   for (auto* x : node->outputs) {
//     if (!role->count(x)) {
//       if (PatternMatch(graph, pat_order, pat_connection, x, pat_order_index + 1,
//                        node_roles, role, false /*reversed*/))
//         return true;
//     }
//   }
//   return false;
// }

}  // namespace details
}  // namespace ir
}  // namespace framework
}  // namespace paddle
