from typing import Tuple

import torch
from torch import nn
  
def CreateMatrix(in, out):
  weights = torch.Tensor(in, out)
  
  # TODO: support init methods:
  return torch.nn.init.xavier_uniform_(nn.Parameter(weights))

def CreateVector(size):
  weights = torch.Tensor(size)
  return torch.nn.init.xavier_uniform_(nn.Parameter(weights))

def ApplyPadding(input, padding, pad_value):
  """
  Args:
    input:   [bs, dim]
    padding: [bs,  1 ],  value 1 is to pad
  """
  return padding*pad_value+input*(1-padding)

class GRUCell(nn.Module):
    """gru cell  https://arxiv.org/pdf/1412.3555.pdf
    theta:
      - w_n: the weight matrix for the input block.
      - w_u: the weight matrix for the update gate
      - w_r: the  eight matrix for the reset gate
      state:
      - m: the GRU output. [batch, output_cell_nodes]
      - c: the GRU cell state. [batch, hidden_cell_nodes]
      inputs:
      - act: a list of input activations. [batch, input_nodes]
      - padding: the padding. [batch, 1].
      - reset_mask: optional 0/1 float input to support packed input training.
        Shape [batch, 1]
    """
    def __init__(self, 
                 num_input_nodes: int,
                 hidden_size: int,
                 num_hidden_nodes: int, 
                 output_size: int = 0,
                 enable_gru_bias: bool = True, 
                 enable_gru_output_bias: bool = True,
                 apply_layer_norm: bool = True,
                 layer_norm_epsilon: float = 1e-8,
                ):
        """Construct an gru cell object."""
        super().__init__()
        
        self.apply_layer_norm = apply_layer_norm
        self.layer_norm_epsilon layer_norm_epsilon 
        
        self.linear_w_n = CreateMatrix(num_input_nodes+output_size, hidden_size)
        self.linear_w_u = CreateMatrix(num_input_nodes+output_size, hidden_size)
        self.linear_w_r = CreateMatrix(num_input_nodes+output_size, output_size)
        
        self.enable_gru_output_bias = enable_gru_output_bias
        if enable_gru_output_bias:
          self.b_n = CreateVector(self.hidden_size)
          self.b_u = CreateVector(self.hidden_size)
          self.b_r = CreateVector(self.output_size)
      
        if output_size:
          self.w_proj = CreateMatrix(hidden_size, output_size)
          if enable_gru_output_bias:
            self.b_proj = CreateVector(output_size)
        
        if apply_layer_norm:
          assert layer_norm_epsilon is None
          self.bn_ln_scale = nn.Parameter(torch.Tensor(hidden_size))
          nn.init.constant_(self.bn_ln_scale, 0)
          
          self.bu_ln_scale = nn.Parameter(torch.Tensor(hidden_size))
          nn.init.constant_(self.bu_ln_scale, 0)
          
          self.br_ln_scale = nn.Parameter(torch.Tensor(output_size))
          nn.init.constant_(self.br_ln_scale, 0)
          
    def forward(
      self, 
      input: torch.Tensor, 
      padding: torch.Tensor,
      state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       """
        Args:
            input (torch.Tensor): [batch, input_nodes).
            padding: [batch, 1]
            state: (m, c)
              m : gru output [batch, output]
              c:  gru cell state [batch, hidden_size]
        Returns:
            state:  (new_m, new_c)
        """
        input_state = torch.concat([input, m], dim=1)
        m, c = state[0], state[1]
        r_g = torch.matmul(input_state, self.linear_w_r)
        
        if self.enable_gru_bias:
          r_g = r_g + self.b_r
        r_g = torch.sigmoid(r_g)
        
        u_g = torch.matmul(input_state, self.linear_w_u)
        n_g = torch.matmul(
          torch.concat([input, torch.mul(r_g, m)], 1), 
          self.linear_fw_n)
        
        # TODO: layer norm here
        if self.enable_gru_bias:
          u_g = u_g + self.b_u
          n_g = n_g + self.b_n
          
          
        u_g = torch.sigmoid(u_g)
        n_g = torch.tanh(n_g)
        new_c = (1.0 - u_g) * (state0.c) + u_g * n_g
        
        # TODO: layer norm here
          
        # TODO clip value here
        
        new_m = torch.matmul(new_c, self.w_proj) + self.b_proj
        
        # apply padding
        new_m = ApplyPadding(new_m, padding, m)
        new_c *= ApplyPadding(new_c, padding, c)
        return (new_m, new_c)
