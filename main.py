import torch

from models.gru_cell import GRUCell
from models.rnn import RNN, StackedRNNLayer
from models.rnnlm import RNNLM, RNNEncoder, init_lm_model

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

xs, state1 = rnn(input, padding, state)

stacked_rnn = StackedRNNLayer('gru', 10, 10, 10, 10)
multi_layer_xs, multi_layer_state1_m, _ = stacked_rnn(input, padding)

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

rnnlm = RNNLM(vocab_size=100, lm_encoder=rnn_encoder, lsm_weight=0.9)
label = torch.ones(10, 20, dtype=torch.int64)
label[:, 10:20] = -1
label_len = torch.ones(10) * 10

loss, ppl, total_ppl, valid_words = rnnlm(input, input_len, label, label_len)
print(loss)
print(ppl)
print(total_ppl)
print(valid_words)

import yaml

with open('test.yaml', 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

configs['vocab_size'] = 100
model = init_lm_model(configs=configs)

_, ppl, _, valid_words = model(input, input_len, label, label_len)
print(ppl.shape)
print(valid_words)

zero_state = model.zero_states(1)
# model.forward_step()
print(zero_state[0].shape)
print(zero_state[1].shape)

a = torch.tensor([[1]], dtype=torch.int)
print("==================")
model.forward_step(a, a, zero_state[0], zero_state[1])
