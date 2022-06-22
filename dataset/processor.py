# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: this code from https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset.py
# and make some changes to support lm way
#
#format1:
#file.list:
#     a.txt
#     b.txt
#     c.txt
#     .....
#     each txt file contains many line,  eg:
#     a.txt:
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     ...
# format2:
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
#     你好wenet lm fusion\t 你 好 _we net _lm _fu sion
# TODO:
# format3: (tar)

import logging
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
from torch.nn.utils.rnn import pad_sequence


def text_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.
        Args:
            data(Iterable[str]): url or local file list
        Returns:
            Iterable[{src, stream}]
    """
    url_opener(data, mode="r")
    for sample in data:
        assert 'src' in sample
        yield sample


# TODO: text_opener and than read from stream1 to support format1


def url_opener(data, mode='rb'):
    """ Give url or local file, return file descriptor
        Inplace operation.
        Args:
            data(Iterable[str]): url or local file list
        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            _ = ex
            logging.warning('Failed to open {}'.format(url))


def parse_raw_text(data):
    """ Parse txt from  line
        (NOTE): for lm
        Args:
            data: Iterable[str], str is a line: 你好 wenet\t你 好 _we net
        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        segs = sample['src'].strip("\n").split("\t")
        if len(segs) != 2:
            continue

        example = dict(txt=segs[0], tokens=segs[1])
        yield example


def filter_text(data, token_max_length=200, token_min_length=1):
    """
    """
    for sample in data:
        assert 'txt' in sample
        assert 'tokens' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        yield sample


def tokenize_space(data, symbol_table):
    """ split tokens to labels
        Inplace operation
        Args:
            data: Iterable[{txt, tokens}]
        Returns:
            Iterable[{txt, tokens, label}]
    """
    for sample in data:
        assert 'txt' in sample
        label = [len(symbol_table) - 1]  # sos
        tokens = sample['tokens'].strip()
        for token in tokens.split(" "):
            if token in symbol_table:
                label.append(symbol_table[token])
            else:
                label.append(symbol_table['<unk>'])
        label.append(len(symbol_table) - 1)  # eos
        sample['label'] = torch.tensor(label, dtype=torch.int32)
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data
        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle
        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort_by_tokens(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`
        Args:
            data: Iterable[{txt, token, label}]
            sort_size: buffer size for sort
        Returns:
            Iterable[{txt, token, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['label'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['label'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`
        Args:
            data: Iterable[txt, token, label}]
            batch_size: batch size
        Returns:
            Iterable[List[{txt, token, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_tokens_in_batch=12000):
    """ Dynamic batch the data until the total tokens in batch
        reach `max_frames_in_batch`
        Args:
            data: Iterable[{txt, tokens, label}]
            max_frames_in_batch: max_frames in one batch
        Returns:
            Iterable[List[{txt, tokens, label}]]
    """
    buf = []
    longest_tokens = 0
    for sample in data:
        assert 'label' in sample
        assert isinstance(sample['label'], torch.Tensor)
        n_tokens = sample['label'].size(0)
        longest_tokens = max(longest_tokens, n_tokens)
        tokens_after_padding = longest_tokens * (len(buf) + 1)
        if tokens_after_padding > max_tokens_in_batch:
            yield buf
            buf = [sample]
            longest_tokens = n_tokens
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_tokens_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_tokens_in_batch=max_tokens_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data
        Args:
            data: Iterable[List[{txt, tokens, label}]]
        Returns:
            Iterable[Tuple(txt, tokens, input, input_lens, label_eos, label_lens)]
    """
    for sample in data:
        assert isinstance(sample, list)
        txt = []
        tokens = []
        input = []
        label_eos = []
        # batch_size = len(sample)

        # sos = torch.ones()
        for x in sample:
            txt.append(x["txt"])
            tokens.append(x["tokens"])
            label = x["label"]
            input.append(label[:-1])  # before eos
            label_eos.append(label[1:])  # after sos

        input = torch.nn.utils.rnn.pad_sequence(
            input, batch_first=True, padding_value=0)  # [batch, seq_len]
        input_lens = torch.where(input != 0, 1, 0).sum(dim=1)  # [batch]
        label_eos = torch.nn.utils.rnn.pad_sequence(
            label_eos,
            batch_first=True,
            padding_value=-1,
        )  # [batch, seq_len]
        label_lens = torch.where(label_eos != -1, 1, 0).sum(dim=1)  #[batch]
        yield (txt, tokens, input, input_lens, label_eos, label_lens)
