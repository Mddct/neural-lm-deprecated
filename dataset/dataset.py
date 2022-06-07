# Note: this code from https://github.com/wenet-e2e/wenet/blob/main/wenet/dataset/dataset.py
# and make some changes to support lm way
import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import dataset.processor as processor


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        torch.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers
            Args:
                data(List): input data list
            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_type,
            data_list_file,
            symbol_table,
            conf,
            partition=True):
    """ Construct dataset from arguments
    """
    assert data_type in ['raw_txt']
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)
    if data_type == 'raw_txt':
        dataset = Processor(dataset, processor.parse_raw)
    else:
        # TODO: support format1 and format2
        pass

    dataset = Processor(dataset, processor.tokenize_space, symbol_table)
    filter_conf = conf.get('filter_text_conf', {})
    dataset = Processor(dataset, processor.filter_text, **filter_conf)

    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort_by_tokens, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding)
    return dataset
