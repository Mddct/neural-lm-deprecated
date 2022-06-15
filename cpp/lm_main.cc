#include <lm.h>
#include <iostream>
#include "torch/torch.h"

int main() {
  std::string model_path = "test.zip";
  RNNLm lm;
  lm.Read(model_path);

  torch::Tensor state_m;
  torch::Tensor state_c;
  lm.Start(&state_m, &state_c);


  for (int i = 0; i < 1; i++) {
    std::cout << lm.Step(state_m, state_c, 0, 0, &state_m, &state_c)
              << std::endl;
  }
}
