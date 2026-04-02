import dataclasses
import itertools
from collections import Counter, defaultdict
import numpy as np

import pytest
import torch
from torch.utils.data._utils.worker import WorkerInfo

from olmo.data.iterable_dataset_mixture import IterableDatasetMixture


@dataclasses.dataclass
class MockWorkerInfo:
    id: int
    num_workers: int


@dataclasses.dataclass
class MockItem:
    dataset: str
    idx: int
    epoch: int


@dataclasses.dataclass
class MockDataset:
    name: str
    n: int

    def __len__(self):
        return self.n

    def get(self, item, epoch=0):
        assert 0 <= item < self.n
        return MockItem(self.name, item, epoch)


def test_single_dataset():
    ds = IterableDatasetMixture(
        [MockDataset("", 10)], global_batch_size=2)
    it = iter(ds)
    for epoch in range(10):
        epoch_data = [next(it) for _ in range(10)]
        assert set(x.idx for x in epoch_data) == set(range(10))
        assert all(x.epoch == epoch for x in epoch_data)


def test_mixture():
    ds = IterableDatasetMixture(
        [MockDataset("a", 10), MockDataset("b", 3)],
        mixture_rates=[0.9, 0.1],
        global_batch_size=2,
        seed=32
    )
    it = iter(ds)
    for_a = []
    for_b = []
    for _ in range(200):
        item: MockItem = next(it)
        if item.dataset == "a":
            for_a.append(item)
        else:
            for_b.append(item)
    assert len(for_a) > (len(for_b) * 0.2)
    for items, ds_len in [(for_a, 10), (for_b, 3)]:
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))


@pytest.mark.parametrize("world_size,num_workers,device_batch_size", [
    (1, 1, 2),
    (3, 1, 4),
    (2, 2, 2),
    (2, 2, 8),
    (3, 6, 4),
])
def test_distributed(world_size, num_workers, device_batch_size):
    start = 0
    global_batch_size = device_batch_size*world_size
    iterators = []
    datasets = [MockDataset("a", 5), MockDataset("b", 11)]
    mixture_rates = [0.8, 0.2]
    device_iterators = []
    bk = torch.utils.data.get_worker_info
    for rank in range(world_size):
        worker_iterators = []
        for worker_id in range(num_workers):
            worker_iterators.append(iter(IterableDatasetMixture(
                datasets, mixture_rates=mixture_rates,
                rank=rank, world_size=world_size,
                worker_info=MockWorkerInfo(worker_id, num_workers),
                global_batch_size=global_batch_size, seed=32, start_index=start)))

        def get_device_batch(_worker_its):
            while True:
                for it in _worker_its:
                    batch = []
                    for _ in range(device_batch_size):
                        batch.append(next(it))
                    yield batch

        device_iterators.append(get_device_batch(worker_iterators))
    torch.utils.data.get_worker_info = bk

    grouped_by_dataset = defaultdict(list)
    for i in range(100):
        global_batch = []
        for it in device_iterators:
            global_batch += next(it)
        global_batch.sort(key=lambda x: x.epoch)
        for ex in global_batch:
            grouped_by_dataset[ex.dataset].append(ex)

    for dataset in datasets:
        items = grouped_by_dataset[dataset.name]
        ds_len = dataset.n
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))


@pytest.mark.parametrize("ns,start_index,world_size,rank", [
     ([11], 16, 1, 0),
     ([8], 201, 1, 0),
     ([17, 27], 50, 1, 0),
     ([5, 7], 19, 2, 1),
     ([5, 7, 3], 27, 2, 0),
     ([5], 1, 4, 2),
     ([5, 7], 19, 3, 2)
])
def test_dataset_start_at(ns, start_index, world_size, rank):
    datasets = [MockDataset("", n) for n in ns]
    global_batch_size = world_size
    ff_ds = IterableDatasetMixture(
        datasets, global_batch_size=global_batch_size, start_index=start_index,
        rank=rank, world_size=world_size
    )
    ff_it = iter(ff_ds)
    ds = IterableDatasetMixture(
        datasets, global_batch_size=global_batch_size, start_index=0,
        rank=rank, world_size=world_size
    )
    it = iter(ds)
    for _ in range(start_index):
        for _ in range(global_batch_size//world_size):
            next(it)
    for _ in range(30):
        assert next(it) == next(ff_it)


def test_stratify():
    datasets = [MockDataset("a", 17), MockDataset("b", 12)]
    ds = IterableDatasetMixture(
        datasets, mixture_rates=[0.8, 0.2], global_batch_size=3, stratify=True)
    it = iter(ds)
    grouped_by_dataset = defaultdict(list)
    for _ in range(10):
        ex = next(it)
        grouped_by_dataset[ex.dataset].append(ex)
    assert len(grouped_by_dataset["a"]) == 8
    assert len(grouped_by_dataset["b"]) == 2

    for _ in range(90):
        ex = next(it)
        grouped_by_dataset[ex.dataset].append(ex)
    assert len(grouped_by_dataset["a"]) == 80
    assert len(grouped_by_dataset["b"]) == 20
    for dataset in datasets:
        items = grouped_by_dataset[dataset.name]
        ds_len = dataset.n
        for epoch in range(len(items)//ds_len):
            epoch_items = items[epoch*ds_len:(epoch+1)*ds_len]
            assert all(x.epoch == epoch for x in epoch_items)
            assert set(x.idx for x in epoch_items) == set(range(ds_len))
