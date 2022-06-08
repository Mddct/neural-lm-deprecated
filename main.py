import torch

from models.gru_cell import GRUCell

cell = GRUCell(
    10,
    10,
    10,
)

m, c = cell.zero_state(10)
input = torch.ones(10, 10)  # [batch, input_nodes]
padding = torch.zeros(10, 1)
state = (m, c)

o, s = cell(input, padding, state)
print(o.shape)
print(s.shape)

print(o)
print(s)
