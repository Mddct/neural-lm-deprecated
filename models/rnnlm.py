from typing import Sequence, Tuple

import torch
from torch import nn

from models.adaptive_softmax import AdaptiveLogSoftmax
from models.label_smoothing import LabelSmoothingLoss
from models.rnn import StackedRNNLayer


class RNNEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        input_nodes: int,
        hidden_nodes: int,
        output_nodes: int,
        cell_type: str = "gru",
        dropout_rate: float = 0.0,
        adaptive_softmax: bool = False,
        cutoffs: Sequence[int] = [],
        div_value: float = 2.0,
        tie_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.lookup_table = nn.Embedding(vocab_size, input_nodes)
        self.stacked_rnn = StackedRNNLayer(cell_type, input_nodes,
                                           hidden_nodes, output_nodes,
                                           n_layers, dropout_rate)
        self.adaptive_softmax = adaptive_softmax
        self.tie_embedding = tie_embedding

        if adaptive_softmax:
            # Note: for compatibility: don't use when adaptive softmax is true
            self.out = nn.Linear(vocab_size, input_nodes)
            # TODO: dropout in adaptive sofmax
            self.log_softmax = AdaptiveLogSoftmax(output_nodes,
                                                  vocab_size,
                                                  cutoffs,
                                                  div_value,
                                                  head_bias=True)
        else:
            if tie_embedding:
                if output_nodes != input_nodes:
                    raise ValueError(
                        'When using the tied flag, input nodes must be equal to outputnodes'
                    )
                self.out = nn.Linear(vocab_size, input_nodes)
                self.out.weight = self.lookup_table.weight
            else:
                self.out = nn.Linear(output_nodes, vocab_size)

            self.log_softmax = nn.LogSoftmax(dim=vocab_size)

    def forward(self, input: torch.Tensor, seq_len: torch.Tensor):
        """
        Args:
            input: [batch, time]
            seq_len: [batch]
        """

        bs = input.size(0)
        # id to embedding
        embeddding = self.lookup_table(input)  # [bs, time, dim]

        max_seq_len = torch.max(seq_len)
        assert max_seq_len == input.size(1)

        ids = torch.arange(0, max_seq_len, 1)  # [bs]
        padding = seq_len.unsqueeze(1) <= ids.unsqueeze(0)  # [bs, max_seq_len]

        padding = padding.transpose(0, 1).unsqueeze(2)  #[time, bs, 1]
        embeddding = embeddding.transpose(0, 1)

        zero_state = self.stacked_rnn.zero_state(bs)
        output = self.stacked_rnn(embeddding, padding, zero_state)
        o, _ = output[0], output[1]

        self.adaptive_softmax = True
        if not self.adaptive_softmax:
            if self.tie_embedding:
                o = self.lookup_table(o)

            o = self.out(o)  #[time, bs, vocab_size]

        o = self.log_softmax(o, dim=2)
        o = o.transpose(0, 1).contiguous()  #[batch, time, vocab_size]
        return o

    def forward_step(
        self, input: torch.Tensor, seq_len: torch.Tensor,
        state_m: torch.Tensor, state_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        embeddding = self.lookup_table(input)  # [bs, time, dim]

        # max_seq_len = torch.max(seq_len)
        # ids = torch.arange(0, max_seq_len, 1).unsqueeze(0)  # [bs]
        # padding = seq_len.unsqueeze(1) < ids  # [bs, max_seq_len]

        # padding = padding.transpose(0, 1).unsqueeze(2)  #[time, bs, 1]
        true_word = torch.ones_like(input).unsqueeze(0)
        false_word = ~true_word
        padding = torch.where(torch.greater_equal(input, 0), true_word,
                              false_word)
        embeddding = embeddding.transpose(0, 1)

        output = self.stacked_rnn(embeddding, padding, (state_m, state_c))
        o, s_m, s_c = output[0], output[1], output[2]

        if not self.adaptive_softmax:
            o = self.out(o)  #[time, bs, vocab_size]

        o = self.log_softmax(o)
        o = o.transpose(0, 1).contiguous()  #[batch, time, vocab_size]
        return o, s_m, s_c


class RNNLM(nn.Module):
    """
    """

    def __init__(
        self,
        vocab_size: int,
        lm_encoder: nn.Module,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        """Construct an gru cell object.
        """
        super().__init__()

        self.model = lm_encoder
        self.length_normalized_loss = length_normalized_loss
        self.criterion = LabelSmoothingLoss(vocab_size, -1, lsm_weight,
                                            length_normalized_loss)

    def forward(self, input: torch.Tensor, input_length: torch.Tensor,
                labels: torch.Tensor, labels_length: torch.Tensor):
        """
        Args:
            input (torch.Tensor):  [batch, time].
            input_len: [bs]
            labels: [bs, time] -1 is ignore id
            labels_length: [bs]
        Returns:
            loss (torch.Tensor): float scalar tensor  Note: before batch average
            ppl (torch.Tensor) : [batch] Note: before batch average
            total_ppl (torch.Tensor) : flaot scalar tensor
        """

        assert (input.shape[0] == input_length.shape[0] == labels.shape[0] ==
                labels_length.shape[0]), (input.shape, input_length.shape,
                                          labels.shape, labels_length.shape)
        # logit after sofmax
        logit = self.model(input, input_length)  #[bs, time_stamp, vocab]
        valid_words = labels_length.sum()
        loss, each_seq_loss_in_batch = self.criterion(logit, labels)
        total_ppl = loss.exp(
        ) if self.length_normalized_loss else loss * input.size(
            0) / valid_words

        ppl = each_seq_loss_in_batch.exp()
        return loss, ppl, total_ppl, valid_words

    @torch.jit.export
    def zero_states(self,
                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m, c = self.model.stacked_rnn.zero_state(batch_size)
        return m, c

    @torch.jit.export
    def forward_step(
        self, input: torch.Tensor, output: torch.Tensor, state_m: torch.Tensor,
        state_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [batch_size, 1]

        Returns:
            output: log  probability, [batch_size, vocab_size]
            state:

        """
        bs = input.size(0)
        seq_len = torch.ones(bs, 1)

        # Note: for one step only
        o, s_m, s_c = self.model.forward_step(input, seq_len, state_m, state_c)
        # score = torch.sum(torch.where(input == output, 1, 0), dim=1)
        return o, s_m, s_c
        # return score, s_m, s_c


def init_lm_model(configs) -> nn.Module:

    vocab_size = configs['vocab_size']
    encoder_type = configs.get('encoder', 'rnn')

    assert encoder_type == 'rnn'
    # cutoffs in yaml: 100,200,400 -> [100,200,400]
    if 'adaptive_softmax' in configs['encoder_conf']:
        assert 'cutoffs' in configs['encoder_conf']
        cutoffs_str = configs['encoder_conf']['cutoffs']
        cutoffs = []
        for cut in cutoffs_str.replace(' ', "", -1).split(","):
            cutoffs.append(int(cut))
        configs['encoder_conf']['cutoffs'] = cutoffs

    encoder = RNNEncoder(vocab_size=vocab_size, **configs['encoder_conf'])

    rnnlm = RNNLM(
        vocab_size=vocab_size,
        lm_encoder=encoder,
        **configs['model_conf'],
    )
    return rnnlm
