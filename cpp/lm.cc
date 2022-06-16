#include "lm.h"

#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

void RNNLm::Read(const std::string& model_path, const int num_threads) {
  torch::jit::script::Module model = torch::jit::load(model_path);
  module_.reset(new TorchModule(std::move(model)));

  at::set_num_threads(num_threads);
}

float RNNLm::Step(torch::Tensor& state_m, torch::Tensor& state_c,
                  int prev_ilabel, int ilabel, torch::Tensor* next_state_m,
                  torch::Tensor* next_state_c) {
  torch::Tensor input = torch::zeros({1, 1}, torch::kInt);
  input[0][0] = prev_ilabel;

  // model input: for label
  torch::Tensor output = torch::zeros({1, 1}, torch::kInt);
  output[0][0] = ilabel;

  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs = {input, output, state_m, state_c};
  auto outputs =
      module_->get_method("forward_step")(inputs).toTuple()->elements();
  *next_state_m = std::move(outputs[1].toTensor());
  *next_state_c = std::move(outputs[2].toTensor());
  return outputs[0].toTensor().accessor<float, 3>()[0][0][ilabel];
}

float RNNLm::StepEos(torch::Tensor& state_m, torch::Tensor& state_c,
                     int prev_ilabel, torch::Tensor* next_state_m,
                     torch::Tensor* next_state_c) {
  return Step(state_m, state_c, prev_ilabel, eos_, next_state_m, next_state_c);
}

void RNNLm::Start(torch::Tensor* next_state_m, torch::Tensor* next_state_c) {
  torch::NoGradGuard no_grad;
  auto output = module_->run_method("zero_states", 1).toTuple()->elements();

  *next_state_m = std::move(output[0].toTensor());
  *next_state_c = std::move(output[1].toTensor());
}
