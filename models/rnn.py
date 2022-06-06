from typing import Tuple

import torch
from torch import nn

from models.gru_cell import RNNCell


class RNN(nn.Module):
    """
    """

    def __init__(self, cell: RNNCell):
        """Construct an gru cell object.
        """
        super().__init__()
        self.cell = cell

    def forward(
        self,
        input: torch.Tensor,
        paddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): [time_stamp, batch, input_nodes).
            padding: [time_stamp, batch, 1]
        Returns:
            output: [time_stamp, batch, output_dim]
            state:  [time_stamp, batch, hidden_dim]
        """
        sequence_length = input.size(0)

        outputs = []
        states = []
        state = self.cell.zero_state()
        for idx in range(sequence_length):
            output, state = self.cell(input[idx], paddings[idx], state)
            outputs.append(output)
            states.append(state)

        return (torch.stack(outputs), torch.stack(states))
