import os
import random
from functools import partial

import torch
import torch.distributed as dist
from kaldiio import ReadHelper
from torch.utils.data import IterableDataset

from funasr.datasets.large_datasets.datapipes.batch import MaxTokenBucketizerIterDataPipe
from funasr.datasets.large_datasets.datapipes.filter import FilterIterDataPipe
from funasr.datasets.large_datasets.datapipes.map import MapperIterDataPipe
from funasr.datasets.large_datasets.utils.filter import filter
from funasr.datasets.large_datasets.utils.padding import padding
from funasr.datasets.large_datasets.utils.tokenize import tokenize


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            parts = line.strip()
            lists.append(parts)
    return lists


class AudioDataset(IterableDataset):
    def __init__(self, scp_lists, data_names, data_types, shuffle=True, mode="train"):
        self.scp_lists = scp_lists
        self.data_names = data_names
        self.data_types = data_types
        self.shuffle = shuffle
        self.mode = mode
        self.epoch = -1
        self.rank = 0
        self.world_size = 1
        self.worker_id = 0
        self.num_workers = 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_rank_data_list(self, data_index):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        if self.mode == "train":
            if self.shuffle:
                random.seed(self.epoch)
                random.shuffle(data_index)
            return data_index[self.rank::self.world_size]

        return data_index

    def get_worker_data_list(self, rank_data_index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        return rank_data_index[self.worker_id::self.num_workers]

    def close_reader(self, reader_list):
        for reader in reader_list:
            reader.close()

    def __iter__(self):
        data_index = list(range(len(self.scp_lists)))
        rank_data_index = self.get_rank_data_list(data_index)
        worker_data_index = self.get_worker_data_list(rank_data_index)

        for index in worker_data_index:
            data = dict(scp=self.scp_lists[index])

            assert 'scp' in data
            scp = data['scp']
            data_file_list = scp.strip().split()
            data_name_list = self.data_names.split(",")
            data_type_list = self.data_types.split(",")

            for file in data_file_list:
                assert os.path.exists(file), "{} not exists".format(file)

            assert len(data_file_list) == len(data_name_list) == len(data_type_list), \
                "The item number of data, data_names, data_types must be the same "

            reader_list = []
            for data_file, data_type in zip(data_file_list, data_type_list):
                if data_type == "kaldi_ark":
                    ark_reader = ReadHelper('ark:{}'.format(data_file))
                    reader_list.append(ark_reader)
                elif data_type == "text":
                    text_reader = open(data_file, "r")
                    reader_list.append(text_reader)
                else:
                    raise TypeError("Data type {} is not supported".format(data_type))

            for items in zip(*reader_list):
                sample_dict = {}
                for item, (data_name, data_type) in zip(items, zip(data_name_list, data_type_list)):
                    if data_type == "kaldi_ark":
                        key, mat = item
                        sample_dict[data_name] = mat
                        if data_name == "speech":
                            sample_dict["key"] = key
                    else:
                        text = item
                        sample_dict[data_name] = text.strip().split()[1:]
                yield sample_dict

            self.close_reader(reader_list)


def len_fn_example(data):
    return len(data)


def len_fn_token(data):
    assert "speech" in data
    return data["speech"].shape[0]


def Dataset(data_list_file,
            dict,
            conf,
            mode="train"):
    scp_lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    data_names = conf.get("data_names", "speech,text")
    data_types = conf.get("data_types", "kaldi_ark,text")
    dataset = AudioDataset(scp_lists, data_names, data_types, shuffle=shuffle, mode=mode)

    filter_conf = conf.get('filter_conf', {})
    filter_fn = partial(filter, **filter_conf)
    dataset = FilterIterDataPipe(dataset, fn=filter_fn)

    vocab = {'vocab': dict}
    tokenize_fn = partial(tokenize, **vocab)
    dataset = MapperIterDataPipe(dataset, fn=tokenize_fn)

    if shuffle:
        buffer_conf = conf.get('shuffle_conf', {})
        buffer_size = buffer_conf['shuffle_size']
        sort_size = buffer_conf['sort_size']
    else:
        buffer_size = 0
        sort_size = 1

    batch_conf = conf.get('batch_conf', {})
    batch_size = batch_conf['batch_size']
    batch_type = batch_conf['batch_type']

    assert batch_type in ["example", "token"]
    if batch_type == 'example':
        len_fn = len_fn_example
    else:
        len_fn = len_fn_token

    dataset = MaxTokenBucketizerIterDataPipe(dataset,
                                             batch_size=batch_size,
                                             len_fn=len_fn,
                                             buffer_size=buffer_size,
                                             sort_size=sort_size)

    dataset = MapperIterDataPipe(dataset, fn=padding)

    return dataset
