#include <gtest/gtest.h>
#include "glog/logging.h"
#include <fstream>
#include "paddle/fluid/inference/api/paddle_inference_api.h"

namespace paddle {
using namespace paddle::contrib;

std::vector<std::string> split(const std::string& line, char delim) {
  std::stringstream ss(line);
  std::vector<std::string> res;

  std::string f;
  while (std::getline(ss, f, delim)) {
    res.push_back(f);
  }

  return res;
}

void PrepareTensor(const std::string& line, PaddleTensor* tensor) {
  std::vector<std::string> fs = split(line, ' ');
  std::vector<float> vs;
  for (auto v : fs) {
    vs.push_back(std::stof(v));
  }

  if (vs.size() != 3)
  vs.pop_back();

  tensor->dtype = PaddleDType::FLOAT32;
  tensor->data.Resize(sizeof(float) * vs.size());
  LOG(INFO) << "Load " << vs.size();
  memcpy(tensor->data.data(), vs.data(), tensor->data.length());
}

void PrepareInputs(const std::string& data_path,
                   std::vector<PaddleTensor>* inputs) {
  std::ifstream file(data_path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  auto& tensor1 = inputs->front();
  auto& tensor2 = inputs->back();
  PrepareTensor(lines[0], &tensor1);
  PrepareTensor(lines[1], &tensor2);

  tensor1.shape.assign({1, 3, 667, 500});
  tensor2.shape.assign({1, 3});
}

TEST(vis, analysis) {
  NativeConfig config;
  config.prog_file =
      "/home/chunwei/Downloads/paddle_bug_fix/paddle_product_det/models/"
      "det_coco_mobilenet/model";
  config.param_file =
      "/home/chunwei/Downloads/paddle_bug_fix/paddle_product_det/models/"
      "det_coco_mobilenet/params";
  config.fraction_of_gpu_memory = 0.1;
  config.specify_input_name = true;

  // Create inputs.
  std::vector<PaddleTensor> inputs(2);
  inputs[0].name = "generate_proposals_0.tmp_0";
  inputs[1].name = "cls_score.tmp_0";

  // Creatre outputs.
  std::vector<PaddleTensor> outputs;

  // config.enable_ir_optim = true;
  // config.pass_builder()->TurnOnDebug();
  auto predictor = CreatePaddlePredictor(config);

  PrepareInputs("/home/chunwei/Downloads/coco_data.txt", &inputs);

  ASSERT_TRUE(predictor->Run(inputs, &outputs));
}

}  // namespace paddle
