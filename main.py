import torch

from models.gru_cell import GRUCell
from models.rnn import RNN, StackedRNNLayer

cell = GRUCell(
    10,
    10,
    10,
)

m, c = cell.zero_state(10)

input = torch.ones(10, 10)  # [batch, input_nodes]
padding = torch.concat([torch.zeros(5, 1), torch.ones(5, 1)], dim=0)
# padding = torch.zeros(10, 1)
state = (m, c)

o, s = cell(input, padding, state)

print(o.shape)
print(s.shape)

rnn = RNN(cell=cell)

input = torch.ones(10, 10, 10)
padding = torch.zeros(10, 10, 1)

xs, state1 = rnn(input, padding)
print(xs.shape)
print(state1[1].shape)

stacked_rnn = StackedRNNLayer('gru', 10, 10, 10, 10)
multi_layer_xs, multi_layer_state1 = stacked_rnn(input, padding)
print(multi_layer_xs.shape)
print(type(multi_layer_state1))
