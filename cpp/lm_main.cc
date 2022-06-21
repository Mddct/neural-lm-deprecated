#include <lm.h>
#include <string>
#include "torch/torch.h"
#include <iostream>

int main() {
  std::string model_path = "test.zip";
  RNNLm lm;
  lm.Read(model_path);

  torch::Tensor state_m;
  torch::Tensor state_c;
  lm.Start(&state_m, &state_c);

  for (int i = 0; i < 1; i++) {
    float score = lm.Step(state_m, state_c, 0, 0, &state_m, &state_c);
    // std::cout << score << std::endl;
  }

  std::cout << 100 << std::endl;
}
