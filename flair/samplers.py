import logging
import pickle
import random
from collections import defaultdict
from typing import Callable, Optional

import boto3
import torch
from torch.utils.data.sampler import Sampler

from flair.data import ConcatFlairDataset

log = logging.getLogger("flair")


class FlairSampler(Sampler):
    def set_dataset(self, data_source):
        """Initialize the data source for the FlairSampler.

        Args:
            data_source: dataset to sample from.
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __len__(self) -> int:
        return self.num_samples


class ImbalancedClassificationDatasetSampler(FlairSampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset."""

    def __init__(self) -> None:
        super().__init__(None)

    def set_dataset(self, data_source):
        """Initialize the dataset used for sampling."""
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = list(range(len(data_source)))

        # first determine the distribution of classes in the dataset
        label_count: dict[str, int] = defaultdict(int)
        for sentence in data_source:
            for label in sentence.labels:
                label_count[label.value] += 1

        # weight for each sample
        offset = 0
        weights = [1.0 / (offset + label_count[data_source[idx].labels[0].value]) for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


class ChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling.

    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, block_size=5, plus_window=5) -> None:
        super().__init__(None)
        self.block_size = block_size
        self.plus_window = plus_window
        self.data_source = None

    def __iter__(self):
        data = list(range(len(self.data_source)))

        blocksize = self.block_size + random.randint(0, self.plus_window)

        log.info(f"Chunk sampling with blocksize = {blocksize} ({self.block_size} + {self.plus_window})")

        # Create blocks
        blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)


class ExpandingChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling.

    Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, step=3) -> None:
        """Initialize the ExpandingChunkSampler.

        Args:
            step: every *step* epochs the block size increments by one.
        """
        super().__init__(None)
        self.block_size = 1
        self.epoch_count = 0
        self.step = step

    def __iter__(self):
        self.epoch_count += 1

        data = list(range(len(self.data_source)))

        log.info(f"Chunk sampling with blocksize = {self.block_size}")

        # Create blocks
        blocks = [data[i : i + self.block_size] for i in range(0, len(data), self.block_size)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        if self.epoch_count % self.step == 0:
            self.block_size += 1

        return iter(data)


class SingleTaskAdaptiveBatchSampler:
    # TODO: docstring
    def __init__(
        self,
        dataset: ConcatFlairDataset,
        batch_size: int = 2,
        downsample_ratio_func: Optional[Callable[[float], float]] = None,
        seed: int = 0,
        debug_mode: bool = False,
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
    ):
        if not isinstance(dataset, ConcatFlairDataset):
            raise RuntimeError("SingleTaskAdaptiveBatchSampler only supports ConcatFlairDataset!")
        assert len(dataset.cummulative_sizes) == len(dataset.ids)
        self.cummulative_sizes = dataset.cummulative_sizes
        self.task_ids = dataset.ids
        log.warning(f"cummulative_sizes: {dataset.cummulative_sizes!s}")
        log.warning(f"task_ids: {dataset.ids!s}")

        self.batch_size = batch_size
        self.downsample_ratio_func = downsample_ratio_func
        self.seed = seed
        self.debug_mode = debug_mode
        self.s3_bucket = s3_bucket
        self.s3_folder = s3_folder

        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

    def generate_batches(self, prev_dev_micro_f1: dict[str, float], shuffle: bool, epoch: int):
        # TODO: docstring
        self.batches = []

        for task_index, task_id in enumerate(self.task_ids):
            start_idx = 0 if task_index == 0 else self.cummulative_sizes[task_index - 1]
            end_idx = self.cummulative_sizes[task_index]  # exclusive
            num_datapoints_this_task = end_idx - start_idx
            indices_this_task = list(range(start_idx, end_idx))

            if self.downsample_ratio_func:
                downsample_ratio = self.downsample_ratio_func(prev_dev_micro_f1[task_id])
                downsample_size = int(num_datapoints_this_task * downsample_ratio)
                log.warning(
                    f"For {task_id}, downsample_ratio is {downsample_ratio:.4f} and downsample_size is {downsample_size}"
                )
                if downsample_size < num_datapoints_this_task:
                    # indices_this_task must be identical across all GPUs
                    random.seed(self.seed + epoch)
                    num_datapoints_this_task = downsample_size
                    indices_this_task = random.sample(indices_this_task, downsample_size)

            if shuffle:
                g = torch.Generator()
                # indices_this_task must be identical across all GPUs
                g.manual_seed(self.seed + epoch)
                perm = torch.randperm(num_datapoints_this_task, generator=g).tolist()
                indices_this_task = [indices_this_task[i] for i in perm]

            # Distribute indices_this_task to each GPU
            num_datapoints_this_task_per_GPU = (
                num_datapoints_this_task // self.num_replicas
            )  # drop the remaining training data if it is not divisible by self.num_replicas
            num_datapoints_this_task_per_GPU = (
                num_datapoints_this_task_per_GPU // self.batch_size * self.batch_size
            )  # drop the remaining training data if it is not divisible by self.batch_size
            indices_this_task_this_GPU = indices_this_task[
                self.rank : num_datapoints_this_task_per_GPU * self.num_replicas : self.num_replicas
            ]
            self.batches += [
                indices_this_task_this_GPU[i : i + self.batch_size]
                for i in range(0, num_datapoints_this_task_per_GPU, self.batch_size)
            ]

        if shuffle:
            g = torch.Generator()
            # perm must be identical across all GPUs
            g.manual_seed(self.seed + epoch)
            perm = torch.randperm(len(self.batches), generator=g).tolist()
            self.batches = [self.batches[i] for i in perm]

        if self.debug_mode:
            boto3.resource("s3").Object(
                self.s3_bucket, f"{self.s3_folder}/batches_rank_{self.rank}_epoch_{epoch}.pkl"
            ).put(Body=pickle.dumps(self.batches))
            log.warning(
                f"Dump batches to s3://{self.s3_bucket}/{self.s3_folder}/batches_rank_{self.rank}_epoch_{epoch}.pkl"
            )

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
