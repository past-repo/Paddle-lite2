#include "paddle/fluid/framework/infer_executor.h"
namespace paddle {
namespace framework {

void InferExecutor::FeedDatas(Scope *scope,
                              const std::map<std::string, const LoDTensor *> &data_map,
                              const std::string &holder_name) {
  for (const auto& item : data_map) {
    FeedData(scope, item.second, holder_name, int idx )

  }

}
void InferExecutor::FetchDatas(Scope *scope,
                               std::map<std::string, LoDTensor *> &data_map,
                               const std::string &holder_name) {

}
void InferExecutor::FeedData(Scope *scope, const LoDTensor &tensor, const std::string &holder_name, int idx) {

}
void InferExecutor::FetchData(Scope *scope, LoDTensor *tensor, const std::string &holder_name, int idx) {

}

}  // namespace framework
}  // namespace paddle
