from pickle import LIST
from typing import List, Optional, Tuple

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

    def zero_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cell.zero_state(batch_size=batch_size)

    def forward(
        self,
        input: torch.Tensor,
        paddings: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): [time, batch, input_nodes).
            padding: [time, batch, 1]
        Returns:
            output: [time, batch, output_dim]
            state:  final state,
                    m: [batch, output_dim]
                    c: [batch, hidden_dim]
        """
        sequence_length = input.size(0)
        batch_size = input.size(1)

        outputs = []
        if state is None:
            state = self.cell.zero_state(batch_size=batch_size)

        state = self.cell.forward(input[0], paddings[0], state)
        outputs.append(state[0])
        for idx in range(sequence_length - 1):
            state = self.cell.forward(input[idx + 1], paddings[idx + 1], state)

            outputs.append(state[0])

        return torch.stack(outputs, dim=0), state


class StackedRNNLayer(nn.Module):

    def __init__(self,
                 cell_type: str = "gru",
                 input_nodes: int = 1024,
                 hidden_nodes: int = 1024,
                 output_nodes: int = 1024,
                 n_layers: int = 1,
                 dropout_rate: float = 0,
                 skip_first_dropout: bool = True) -> None:

        super().__init__()

        # List not ModuleList
        # self,
        # num_input_nodes: int,
        # hidden_size: int,
        # output_size: int = 0,
        # enable_gru_output_bias: bool = True,
        # apply_layer_norm: bool = True,
        # layer_norm_epsilon: float = 1e-8,

        self.rnn = nn.ModuleList()
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
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # whether to skip first layer
        self.skip_first_dropout = skip_first_dropout

    def zero_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:

        states_m: List[torch.Tensor] = []
        states_c: List[torch.Tensor] = []
        for layer in self.rnn:
            state = layer.zero_state(batch_size)
            states_m.append(state[0].unsqueeze(0))
            states_c.append(state[1].unsqueeze(0))

        # [nlayer, bs, 1, dim]

        return torch.concat(states_m, dim=0), torch.concat(states_c, dim=0)

    def forward(
        self,
        input: torch.Tensor,
        padding: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [time, batch, dim]
            padding: [time, batch, 1]

        Returns:
            output: [time, batch, output_size]
            state1: final state each layer
        """
        batch_size = input.size(1)
        if states is None:
            state0 = self.zero_state(batch_size=batch_size)
        else:
            state0 = states

        xs = input

        if not self.skip_first_dropout:
            xs = self.dropout(xs)

        states1_m = []
        states1_c = []
        for (i, layer) in enumerate(self.rnn):
            # ys, s = self.rnn[i](xs, padding, state0[i])
            ys, s = layer(xs, padding, (state0[0][i], state0[1][i]))
            ys = self.dropout(ys)
            states1_m.append(s[0].unsqueeze(0))
            states1_c.append(s[1].unsqueeze(0))
            # rnn base have embedding , different input/output shape
            # internal layer have same input output shape:
            # 1->ouput+2->output ...
            # TODO: qadd
            if i >= 1 and i <= self.n_layers - 1:
                xs = xs + ys
            else:
                xs = ys

        return xs, torch.concat(states1_m, dim=0), torch.concat(states1_c,
                                                                dim=0)
