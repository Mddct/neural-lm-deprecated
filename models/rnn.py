from models.gru_cell import RNNCell

import torch

class RNN(nn.Module):
    """
    """
    def __init__(
      self, 
      cell: RNNCell):
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
            output: [time_stamp, output_dim]
            state:  [time_stamp, batch, hidden_dim]
      """
      sequence_length = input.size(0)
      
      outputs = [None] * sequence_length
      states = [None] * sequence_length
      state = self.cell.zero_state()
      for idx in range(sequence_length):
        output, state = self.cell(input[idx], paddings[i], state)
        outputs[i] = output
        states[i] = state
        
      outputs = torch.stack(outputs, dim=1)
      states = torch.stack(states, dim=1)
      return (outputs, states)
