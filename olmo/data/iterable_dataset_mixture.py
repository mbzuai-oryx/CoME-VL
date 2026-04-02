import dataclasses
import logging
from typing import List, Optional, Any, Dict

import numpy as np
import torch
from torch.utils.data import Sampler

from olmo.data.dataset import DeterministicDataset
from olmo.torch_util import get_world_size, get_global_rank

log = logging.getLogger(__name__)


class IterableDatasetMixture(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Infinitely iterates over a mixture of datasets"""

    def __init__(
        self,
        datasets: List[DeterministicDataset],
        global_batch_size: int,
        mixture_rates: List[float]=None,
        seed: int = 0,
        start_index: int = 0,
        shuffle: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        stratify: bool = False,
        worker_info=None
    ):
        self.datasets = list(datasets)
        if mixture_rates:
            self.mixture_rates = np.array(mixture_rates, dtype=np.float32)
        else:
            self.mixture_rates = None

        self.seed = seed
        assert seed is not None
        self.start_index = start_index
        self.shuffle = shuffle
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_global_rank()
        self.global_batch_size = global_batch_size
        assert self.global_batch_size % self.world_size == 0
        self.device_batch_size = global_batch_size // self.world_size
        self.stratify = stratify
        self.worker_info = worker_info  # For testing

    def _get_next_sources(self, rng, counts):
        if len(self.datasets) == 1:
            return np.zeros(self.global_batch_size, dtype=np.int32)
        if self.stratify:
            out = []
            counts = np.copy(counts)
            total = counts.sum()
            for _ in range(self.global_batch_size):
                # Sample the most under-represented dataset
                ix = np.argmax(np.abs(counts/total - self.mixture_rates))
                out.append(ix)
                counts[ix] += 1
                total += 1
            return np.array(out)
        else:
            return rng.choice(
                len(self.datasets),
                size=self.global_batch_size,
                p=self.mixture_rates
            )

    def __iter__(self):
        worker_info = self.worker_info or torch.utils.data.get_worker_info()
        batch_ix = 0
        rng = np.random.RandomState(self.seed)

        # How often each dataset has been sampled globally across all devices/workers
        counts = np.zeros(len(self.datasets), dtype=np.int64)
        if self.start_index != 0:
            assert self.start_index % self.global_batch_size == 0
            start_batch = self.start_index // self.global_batch_size
            if worker_info is None:
                log.info(f"Fast forwarding instance {self.start_index}, batch {start_batch}...")
            for i in range(start_batch):
                ix = self._get_next_sources(rng, counts)
                batch_ix += 1
                np.add.at(counts, ix, 1)
            if worker_info is None:
                log.info(f"Done")
        shuffled_ixs = [(None, None) for _ in self.datasets]

        while True:
            ix = self._get_next_sources(rng, counts)
            if worker_info and batch_ix % worker_info.num_workers != worker_info.id:
                # Workers participate in every num_workers-th batch, `DataLoader` collects complete
                # batches from individual workers one-by-one so this ensures the number
                # of workers does not affect the order of the data
                np.add.at(counts, ix, 1)
                batch_ix += 1
                continue

            batch_ix += 1
            for i, dataset_ix in enumerate(ix):
                count = counts[dataset_ix]
                counts[dataset_ix] += 1

                if (i + self.rank) % self.world_size != 0:
                    continue
                device_ix = (i + self.rank) // self.world_size

                dataset = self.datasets[dataset_ix]
                epoch = count // len(dataset)

                shuffled_for, shuffled_order = shuffled_ixs[dataset_ix]
                if epoch != shuffled_for:
                    shuffle_seed = self.seed + epoch * 1771
                    shuffled_order = np.arange(len(dataset), dtype=np.int32)
                    np.random.RandomState(shuffle_seed).shuffle(shuffled_order)
                    shuffled_ixs[dataset_ix] = (epoch, shuffled_order)

                yield dataset.get(int(shuffled_order[count % len(dataset)]), epoch)
