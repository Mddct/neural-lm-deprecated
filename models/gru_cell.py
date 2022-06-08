from typing import Tuple

import torch
from torch import nn
from torch.functional import Tensor


class RNNCell(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        padding: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return NotImplemented("abstract method")

    def zero_state(self, batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = batch_size
        return NotImplemented("abstract method")


def zero_state_helper(batch_size,
                      hidden,
                      output,
                      method="zero") -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: more methods
    _ = method
    return torch.zeros(batch_size, hidden), torch.zeros(batch_size, output)


def CreateMatrix(first, second):
    weights = torch.Tensor(first, second)

    # TODO: support init methods:
    return torch.nn.init.xavier_uniform_(nn.parameter.Parameter(weights))


def CreateVector(size):
    weights = torch.Tensor(size)
    return torch.nn.init.zeros_(nn.parameter.Parameter(weights))


def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
     Args:
        input:   [bs, dim]
        padding: [bs,  1 ],  value 1 is to pad
    """
    # return padding * pad_value + input * (1 - padding)
    return padding * pad_value + input * (~padding)


class GRUCell(RNNCell):
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

    def __init__(
        self,
        num_input_nodes: int,
        hidden_size: int,
        output_size: int = 0,
        enable_gru_output_bias: bool = True,
        apply_layer_norm: bool = True,
        layer_norm_epsilon: float = 1e-8,
    ):
        """Construct an gru cell object."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_input_nodes = num_input_nodes
        self.output_size = output_size

        self.apply_layer_norm = apply_layer_norm
        self.layer_norm_epsilon = layer_norm_epsilon

        self.linear_w_n = CreateMatrix(num_input_nodes + output_size,
                                       hidden_size)
        self.linear_w_u = CreateMatrix(num_input_nodes + output_size,
                                       hidden_size)
        self.linear_w_r = CreateMatrix(num_input_nodes + output_size,
                                       output_size)

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
            assert layer_norm_epsilon is not None
            self.bn_ln_scale = nn.parameter.Parameter(
                torch.Tensor(hidden_size))
            nn.init.zeros_(self.bn_ln_scale)

            self.bu_ln_scale = nn.parameter.Parameter(
                torch.Tensor(hidden_size))
            nn.init.zeros_(self.bu_ln_scale)

            self.br_ln_scale = nn.parameter.Parameter(
                torch.Tensor(output_size))
            nn.init.zeros_(self.br_ln_scale)

    def zero_state(self,
                   batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return zero_state_helper(batch_size, self.hidden_size,
                                 self.output_size)

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
        m, c = state[0], state[1]
        input_state = torch.concat([input, m], dim=1)
        r_g = torch.matmul(input_state, self.linear_w_r)

        if self.enable_gru_output_bias:
            r_g = r_g + self.b_r
            r_g = torch.sigmoid(r_g)

        u_g = torch.matmul(input_state, self.linear_w_u)
        n_g = torch.matmul(torch.concat([input, torch.mul(r_g, m)], 1),
                           self.linear_w_n)

        # TODO: layer norm here
        if self.enable_gru_output_bias:
            u_g = u_g + self.b_u
            n_g = n_g + self.b_n

        u_g = torch.sigmoid(u_g)
        n_g = torch.tanh(n_g)
        new_c = (1.0 - u_g) * (c) + u_g * n_g

        # TODO: layer norm here

        # TODO clip value here

        new_m = torch.matmul(new_c, self.w_proj) + self.b_proj

        # apply padding
        new_m = ApplyPadding(new_m, padding, m)
        new_c = ApplyPadding(new_c, padding, c)

        return new_m, new_c
