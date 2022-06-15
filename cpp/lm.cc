#include "lm.h"


#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

void RNNLm::Read(const std::string& model_path, const int num_threads){
  torch::jit::script::Module model = torch::jit::load(model_path);
  module_.reset(new TorchModule(std::move(model)));

  at::set_num_threads(num_threads);
}


float RNNLm::Step(torch::Tensor& state, int prev_ilabel, int ilabel, torch::Tensor* next_state){
  torch::Tensor input =
      torch::zeros({1, 1}, torch::kInt);
  input[0][0] = prev_ilabel;

  // model input: for label
  torch::Tensor output =
      torch::zeros({1, 1}, torch::kInt);
  output[0][0] = ilabel;


  std::vector<torch::jit::IValue> inputs = {input, state};
  auto outputs = module_->get_method("forward_step")(inputs).toTuple()->elements();
  *next_state = std::move(outputs[1].toTensor());
  return std::move(outputs[0]).toDouble();
}

float RNNLm::StepEos(torch::Tensor& state, int prev_ilabel, torch::Tensor* next_state){
  return Step(state, prev_ilabel, eos_,  next_state);
}

