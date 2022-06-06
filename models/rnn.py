from typing import List, Tuple

import torch
from torch import nn

from models.gru_cell import GRUCell, RNNCell


class RNN(nn.Module):
    """
    """

    def __init__(self, cell: RNNCell):
        """Construct an gru cell object.
        """
        super().__init__()
        self.cell = cell

    def zero_state(self, batch_size):
        return self.cell.zero_state(batch_size=batch_size)

    def forward(
        self,
        input: torch.Tensor,
        paddings: torch.Tensor,
        state: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): [time_stamp, batch, input_nodes).
            padding: [time_stamp, batch, 1]
        Returns:
            output: [time_stamp, batch, output_dim]
            state:  [time_stamp, batch, hidden_dim]
        """
        sequence_length = input.size(0)
        batch_size = input.size(1)

        outputs = []
        if state is None:
            state = self.cell.zero_state(batch_size=batch_size)

        for idx in range(sequence_length):
            output, state = self.cell(input[idx], paddings[idx], state)
            outputs.append(output)

        return (torch.stack(outputs), (outputs[-1], state))


class StackedRNNLayer(nn.Module):

    def __init__(self,
                 cell_type: str = "gru",
                 input_nodes: int = 1024,
                 hidden_nodes: int = 1024,
                 output_nodes: int = 1024,
                 n_layers: int = 1,
                 dropout_rate: float = 0) -> None:

        super().__init__()

        # List not ModuleList
        # self,
        # num_input_nodes: int,
        # hidden_size: int,
        # output_size: int = 0,
        # enable_gru_output_bias: bool = True,
        # apply_layer_norm: bool = True,
        # layer_norm_epsilon: float = 1e-8,

        self.rnn = []
        self.n_layers = n_layers

        # TODO: other cell
        _ = cell_type
        for i in range(n_layers):
            if i == 0:
                cell = GRUCell(input_nodes, hidden_nodes, output_nodes)
            else:
                cell = GRUCell(output_nodes, hidden_nodes, output_nodes)

            self.rnn.append(RNN(cell=cell))

        # dropout rate for each layer
        self.dropout_rate = dropout_rate

    def zero_state(self, batch_size) -> List[torch.Tensor]:

        states = []
        for i in range(self.n_layers):
            zero_state = self.rnn[i].zero_state(batch_size)
            states.append(zero_state)

        return states

    def forward(
        self, input: torch.Tensor, padding: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = input.size(1)
        state0 = self.zero_state(batch_size=batch_size)

        xs = input
        state1 = []
        for i in range(self.n_layers):
            ys, s = self.rnn[i](xs, padding, state0[i])
            state1.append(s)
            # TODO: dropout here
            # TODO: skip layer(residual)
            xs = ys
        return (xs, state1)
