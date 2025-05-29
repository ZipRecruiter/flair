import logging
import pickle
import random
from collections import defaultdict
from collections.abc import Iterator
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


class AdaptiveBatchSamplerBaseClass:
    """Base class for adaptive batch samplers.

    This class provides common functionality for adaptive batch sampling, including:
    - Downsampling data points based on a ratio.
    - Distributing data points across multiple GPUs.
    - Chunking data points into batches.
    - Shuffling batches.
    - Saving batches to S3 for debugging.
    """

    def __init__(
        self,
        seed: int = 0,
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
    ) -> None:
        """Initialize the base sampler.

        Args:
            seed (int): Random seed for reproducibility.
            s3_bucket (Optional[str]): S3 bucket name for saving batches.
            s3_folder (Optional[str]): S3 folder path for saving batches.
        """
        self.seed = seed
        self.s3_bucket = s3_bucket
        self.s3_folder = s3_folder

        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

    def generate_batches(self) -> None:
        """Abstract method to generate batches. Must be implemented by subclasses."""
        raise NotImplementedError

    def downsample_distribute_and_chunk_indices(
        self, indices: list[int], downsample_ratio: Optional[float], shuffle: bool, epoch: int, batch_size: int
    ) -> list[list[int]]:
        """Downsample, distribute, and chunk indices into batches.

        Args:
            indices (list[int]): List of indices, where each index corresponds to a data point in the dataset, e.g., 0 corresponds to the 0th data point.
            downsample_ratio (Optional[float]): Ratio for downsampling data points.
            shuffle (bool): Whether to shuffle the indices.
            epoch (int): Current epoch number for randomness.
            batch_size (int): Size of each batch.

        Returns:
            list[list[int]]: List of batches, where each batch is a list of indices.
        """
        num_datapoints = len(indices)

        # downsample indices
        if downsample_ratio:
            downsample_size = int(num_datapoints * downsample_ratio)
            if downsample_size <= num_datapoints:
                # control randomness to ensure indices are identical across all GPUs
                random.seed(self.seed + epoch)
                indices = random.sample(indices, downsample_size)
                num_datapoints = downsample_size

        if shuffle:
            # control randomness to ensure indices are identical across all GPUs
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            perm = torch.randperm(num_datapoints, generator=g).tolist()
            indices = [indices[i] for i in perm]

        # Distribute indices to each GPU
        num_datapoints_per_GPU = (
            num_datapoints // self.num_replicas
        )  # drop the remaining training data if it is not divisible by self.num_replicas
        num_datapoints_per_GPU = (
            num_datapoints_per_GPU // batch_size * batch_size
        )  # drop the remaining training data if it is not divisible by batch_size
        indices_this_GPU = indices[self.rank : num_datapoints_per_GPU * self.num_replicas : self.num_replicas]

        # chunk indices to batches
        return [indices_this_GPU[i : i + batch_size] for i in range(0, num_datapoints_per_GPU, batch_size)]

    def shuffle_batches(self, batches: list[list[int]], epoch: int) -> list[list[int]]:
        """Shuffle batches.

        Args:
            batches (list[list[int]]): List of batches to shuffle.
            epoch (int): Current epoch number for randomness.

        Returns:
            list[list[int]]: Shuffled list of batches.
        """
        # control randomness to ensure the permutation to shuffle batches are identical across all GPUs
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        perm = torch.randperm(len(batches), generator=g).tolist()
        return [batches[i] for i in perm]

    def save_batches_to_s3(self, epoch: int) -> None:
        """Save batches to S3 for debugging.

        Args:
            epoch (int): Current epoch number for naming the saved file.
        """
        boto3.resource("s3").Object(self.s3_bucket, f"{self.s3_folder}/batches_rank_{self.rank}_epoch_{epoch}.pkl").put(
            Body=pickle.dumps(self.batches)
        )
        log.info(f"Dump batches to s3://{self.s3_bucket}/{self.s3_folder}/batches_rank_{self.rank}_epoch_{epoch}.pkl")

    def __iter__(self) -> Iterator[list[int]]:
        """Return an iterator over the batches."""
        return iter(self.batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batches)


class SingleTaskAdaptiveBatchSampler(AdaptiveBatchSamplerBaseClass):
    """This sampler is designed to generate batches for the training set of a MultiCorpus.

    It guarantees:
    1. On a single CPU/GPU, each batch contains data points from the same task.
    2. On multiple GPUs, all batches processed simultaneously across GPUs contain data points from the same task

    Additionally, this sampler supports adaptive downsampling of training data for each task based on a user-provided function.
    """

    def __init__(
        self,
        dataset: ConcatFlairDataset,
        batch_size: int = 2,
        downsample_ratio_func: Optional[Callable[[float], float]] = None,
        seed: int = 0,
        debug_mode: bool = False,
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset (ConcatFlairDataset): The dataset to sample from.
            batch_size (int): Size of each batch.
            downsample_ratio_func (Optional[Callable[[float], float]]): Function to compute downsample ratio.
            seed (int): Random seed for reproducibility.
            debug_mode (bool): Whether to enable debug mode.
            s3_bucket (Optional[str]): S3 bucket name for saving batches.
            s3_folder (Optional[str]): S3 folder path for saving batches.
        """
        if not isinstance(dataset, ConcatFlairDataset):
            raise RuntimeError("SingleTaskAdaptiveBatchSampler only supports ConcatFlairDataset!")
        assert len(dataset.cummulative_sizes) == len(dataset.ids)
        # cummulative_sizes is an attribute of ConcatFlairDataset that defines the boundaries of each dataset within the concatenated dataset.
        # For example, if cummulative_sizes = [5, 20, 50]:
        # - Data points from the first task start at index 0 and end at index 5 (exclusive) in the ConcatFlairDataset.
        # - Data points from the second task start at index 5 and end at index 20 (exclusive).
        # - Data points from the third task start at index 20 and end at index 50 (exclusive).
        self.cummulative_sizes = dataset.cummulative_sizes
        self.task_ids = dataset.ids

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

        if self.debug_mode:
            log.info(f"cummulative_sizes: {dataset.cummulative_sizes!s}")
            log.info(f"task_ids: {dataset.ids!s}")

    def generate_batches(self, prev_dev_micro_f1: dict[str, float], shuffle: bool, epoch: int) -> None:
        """Generate batches for the dataset.

        Args:
            prev_dev_micro_f1 (dict[str, float]): Micro F1 score on dev set from the previous epoch for each task.
            shuffle (bool): Whether to shuffle the batches.
            epoch (int): Current epoch number for randomness.
        """
        self.batches = []

        for task_index, task_id in enumerate(self.task_ids):
            start_idx = 0 if task_index == 0 else self.cummulative_sizes[task_index - 1]
            end_idx = self.cummulative_sizes[task_index]  # exclusive
            indices_this_task = list(range(start_idx, end_idx))
            downsample_ratio = None
            if self.downsample_ratio_func:
                downsample_ratio = self.downsample_ratio_func(prev_dev_micro_f1[task_id])
                log.info(f"For {task_id}, downsample_ratio is {downsample_ratio:.4f}")

            self.batches += self.downsample_distribute_and_chunk_indices(
                indices=indices_this_task,
                downsample_ratio=downsample_ratio,
                shuffle=shuffle,
                epoch=epoch,
                batch_size=self.batch_size,
            )

        if shuffle:
            self.batches = self.shuffle_batches(self.batches, epoch)

        if self.debug_mode:
            self.save_batches_to_s3(epoch)


class SingleTaskSingleLengthAdaptiveBatchSampler(AdaptiveBatchSamplerBaseClass):
    """This sampler is designed to generate batches for the training set of a MultiCorpus.

    It guarantees:
    1. On a single CPU/GPU, each batch contains data points from the same task and with the same number of tokens.
    2. On multiple GPUs, all batches processed simultaneously across GPUs contain data points from the same task and with the same number of tokens.

    Additionally, this sampler supports adaptive downsampling of training data for each task based on a user-provided function.
    """

    def __init__(
        self,
        dataset: ConcatFlairDataset,
        length_dicts_for_tasks: dict[str, dict[int, list[int]]],
        batch_size_func: Callable[[int], int],
        downsample_ratio_func: Optional[Callable[[float], float]] = None,
        seed: int = 0,
        debug_mode: bool = False,
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset (ConcatFlairDataset): The dataset to sample from.
            length_dicts_for_tasks (dict[str, dict[int, list[int]]]): A dictionary where each key is a task_id, and the value is a dictionary, which maps a length to a list of indices corresponding to data points of that length.
                For example:
                {
                    "task_1": {10: [0, 2], 20: [1]},
                    "task_2": {15: [0], 25: [1, 3], 60: [2, 4]}
                }
                This indicates that for task_1, there are data points of length 10 at indices [0, 2] and of length 20 at indices [1].
                Similarly, for task_2, there are data points of length 15 at indices [0] and of length 25 at indices [1, 3] and of length 60 at indices [2, 4].
            batch_size_func (Callable[[int], int]): Function to compute batch size based on length of data points.
            downsample_ratio_func (Optional[Callable[[float], float]]): Function to compute downsample ratio.
            seed (int): Random seed for reproducibility.
            debug_mode (bool): Whether to enable debug mode for logging and saving batches.
            s3_bucket (Optional[str]): S3 bucket name for saving batches.
            s3_folder (Optional[str]): S3 folder path for saving batches.
        """
        if not isinstance(dataset, ConcatFlairDataset):
            raise RuntimeError("SingleTaskAdaptiveBatchSampler only supports ConcatFlairDataset!")
        assert len(dataset.cummulative_sizes) == len(dataset.ids)
        # cummulative_sizes is an attribute of ConcatFlairDataset that defines the boundaries of each dataset within the concatenated dataset.
        # For example, if cummulative_sizes = [5, 20, 50]:
        # - Data points from the first task start at index 0 and end at index 5 (exclusive) in the ConcatFlairDataset.
        # - Data points from the second task start at index 5 and end at index 20 (exclusive).
        # - Data points from the third task start at index 20 and end at index 50 (exclusive).
        self.cummulative_sizes = dataset.cummulative_sizes
        self.task_ids = dataset.ids

        # Adjust the indices in the length_dict of each task by adding the task's starting index in the concatenated dataset.
        task_start_idx = {}
        task_end_idx = {}
        for task_idx, task_id in enumerate(self.task_ids):
            task_start_idx[task_id] = 0 if task_idx == 0 else self.cummulative_sizes[task_idx - 1]
            task_end_idx[task_id] = self.cummulative_sizes[task_idx]  # exclusive
        self.length_dicts_for_tasks = {}
        for task_id in self.task_ids:
            self.length_dicts_for_tasks[task_id] = {}
            for length, indices in length_dicts_for_tasks[task_id].items():
                self.length_dicts_for_tasks[task_id][length] = (
                    SingleTaskSingleLengthAdaptiveBatchSampler.apply_offset_to_list(indices, task_start_idx[task_id])
                )
                assert max(self.length_dicts_for_tasks[task_id][length]) < task_end_idx[task_id]

        self.batch_size_func = batch_size_func
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

        if self.debug_mode:
            log.info(f"cummulative_sizes: {dataset.cummulative_sizes!s}")
            log.info(f"task_ids: {dataset.ids!s}")

    @staticmethod
    def apply_offset_to_list(list_of_int: list[int], offset: int) -> list[int]:
        """Apply an offset to a list of integers.

        Args:
            list_of_int (list[int]): List of integers.
            offset (int): Offset to apply.

        Returns:
            list[int]: List of integers with the offset applied.
        """
        return [x + offset for x in list_of_int]

    def generate_batches(self, prev_dev_micro_f1: dict[str, float], shuffle: bool, epoch: int) -> None:
        """Generate batches for the dataset.

        Args:
            prev_dev_micro_f1 (dict[str, float]): Micro F1 score on dev set from the previous epoch for each task.
            shuffle (bool): Whether to shuffle the batches.
            epoch (int): Current epoch number for randomness.
        """
        self.batches = []

        for task_id in self.task_ids:
            downsample_ratio = None
            if self.downsample_ratio_func:
                downsample_ratio = self.downsample_ratio_func(prev_dev_micro_f1[task_id])
                log.info(f"For {task_id}, downsample_ratio is {downsample_ratio:.4f}")

            length_dict_this_task = self.length_dicts_for_tasks[task_id]
            for length, indices_this_task_this_length in length_dict_this_task.items():
                batch_size = self.batch_size_func(length)
                self.batches += self.downsample_distribute_and_chunk_indices(
                    indices=indices_this_task_this_length,
                    downsample_ratio=downsample_ratio,
                    shuffle=shuffle,
                    epoch=epoch,
                    batch_size=batch_size,
                )

        if shuffle:
            self.batches = self.shuffle_batches(self.batches, epoch)

        if self.debug_mode:
            self.save_batches_to_s3(epoch)
