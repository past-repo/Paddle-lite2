namespace paddle {
namespace lite {
namespace kernels {
namespace host {

class TanhCompute final : public OpKernel<TARGET(kHost), PRECISION(kFloat)> {
 public:
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
