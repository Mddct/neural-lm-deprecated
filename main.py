import torch

from models.gru_cell import GRUCell
from models.rnn import RNN, StackedRNNLayer
from models.rnnlm import RNNEncoder

cell = GRUCell(
    10,
    10,
    10,
)

m, c = cell.zero_state(10)

input = torch.ones(10, 10)  # [batch, input_nodes]
padding = torch.concat([
    torch.zeros(5, 1, dtype=torch.int32),
    torch.ones(5, 1, dtype=torch.int32)
],
                       dim=0)
# padding = torch.zeros(10, 1)
state = (m, c)

o, s = cell(input, padding, state)

rnn = RNN(cell=cell)

input = torch.ones(10, 10, 10)
padding = torch.zeros(10, 10, 1, dtype=torch.int32)

xs, state1 = rnn(input, padding)

stacked_rnn = StackedRNNLayer('gru', 10, 10, 10, 10)
multi_layer_xs, multi_layer_state1 = stacked_rnn(input, padding)

rnn_encoder = RNNEncoder(vocab_size=100,
                         n_layers=4,
                         input_nodes=10,
                         hidden_nodes=10,
                         output_nodes=10,
                         adaptive_softmax=True,
                         cutoffs=[10, 20])
input = torch.ones(10, 20, dtype=torch.int64)
input_len = torch.ones(10)
input_len[0] = 20
loss = rnn_encoder(input, input_len)
