# -*- coding: utf-8 -*-

from typing import Sequence

import torch
from torch.nn.functional import log_softmax


class AdaptiveLogSoftmax(torch.nn.Module):
    # in_features: int
    # n_classes: int
    # cutoffs: List[int]
    # div_value: float
    # head_bias: bool
    # head: Linear
    # tail: ModuleList

    def __init__(self,
                 in_features: int,
                 n_classes: int,
                 cutoffs: Sequence[int],
                 div_value: float = 4.,
                 head_bias: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaptiveLogSoftmax, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]

        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = torch.nn.Linear(self.in_features,
                                    self.head_size,
                                    bias=self.head_bias,
                                    **factory_kwargs)
        self.tail = torch.nn.ModuleList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value**(2)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = torch.nn.Sequential(
                torch.nn.Linear(self.in_features,
                                hsz,
                                bias=False,
                                **factory_kwargs),
                torch.nn.Linear(hsz, osz, bias=False, **factory_kwargs),
            )

            self.tail.append(projection)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # check shape: [bs, time, vocab]
        assert len(input.size()) == 3
        bs, seq_len = input.size(0), input.size(1)
        input = input.view(bs * seq_len, -1)
        return self.log_prob(input).view(bs, seq_len)

    def _get_full_log_prob_v2(self, input, head_output):
        head_logprob = log_softmax(head_output, dim=1)
        tail_output = []
        tail_output.append(head_logprob[:, :self.shortlist_size])
        for i, tail_layer in enumerate(self.tail):
            cluster_output = tail_layer(input)
            cluster_output = log_softmax(cluster_output, dim=1)

            cluster_output = cluster_output + head_logprob[:, (
                self.shortlist_size + i):(self.shortlist_size + i + 1)]
            tail_output.append(cluster_output)

        return torch.cat(tail_output, dim=1)

    def log_prob(self, input: torch.Tensor) -> torch.Tensor:
        r""" Computes log probabilities for all :math:`\texttt{n\_classes}`

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """

        head_output = self.head(input)
        return self._get_full_log_prob_v2(input, head_output)
