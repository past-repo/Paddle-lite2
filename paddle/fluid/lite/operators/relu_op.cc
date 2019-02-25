#include "paddle/fluid/lite/operators/relu_op.h"
#include "paddle/fluid/lite/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReluOp::CheckShape() const { return true; }
bool ReluOp::InferShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  // TODO(Superjomn) Enable data sharing.
  param_.output->Resize(param_.input->dims());
  // param_.output->ShareDataWith(*param_.input);
  // share lod
  // param_.output->set_lod(param_.input->lod());
  return true;
}

bool ReluOp::Run() { return false; }

bool ReluOp::Build(const framework::OpDesc &opdesc, framework::Scope *scope) {
  return false;
}

REGISTER_LITE_OP(relu, ReluOp);

}  // namespace operators
}  // namespace lite
}  // namespace paddle
