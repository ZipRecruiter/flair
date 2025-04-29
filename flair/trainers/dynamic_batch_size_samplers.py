from collections import defaultdict
from torch.utils.data import BatchSampler
import torch.distributed
import logging
import pickle
import torch
import boto3
import random
from typing import Optional
from transformers import AutoTokenizer

logger = logging.getLogger("flair")


class SingleLengthSingleTaskBatchSampler(BatchSampler):
    def __init__(
        self, 
        max_tokens_per_batch_step: int = 4096,
        max_sentences_per_batch_step: int = 128,
        min_sentences_per_batch_step: int = 2,
        skip_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.max_tokens_per_batch_step = max_tokens_per_batch_step
        self.max_sentences_per_batch_step = max_sentences_per_batch_step
        self.min_sentences_per_batch_step = min_sentences_per_batch_step
        self.skip_ratio = skip_ratio
        logger.warning(f"max_tokens_per_batch_step: {max_tokens_per_batch_step}")
        logger.warning(f"max_sentences_per_batch_step: {max_sentences_per_batch_step}")
        logger.warning(f"min_sentences_per_batch_step: {min_sentences_per_batch_step}")
        logger.warning(f"skip_ratio: {skip_ratio}")
        self.shuffle = shuffle
        self.seed = seed
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0
        self.batches = []

    def set_dataset(self, dataset, sampler_cache_s3_bucket, sampler_cache_s3_folder):
        self.dataset = dataset
        self.cummulative_sizes = dataset.cummulative_sizes
        self.num_tasks = len(self.cummulative_sizes)
        self.sampler_cache_s3_bucket = sampler_cache_s3_bucket
        self.sampler_cache_s3_folder = sampler_cache_s3_folder
        cache_file_name = f"{sampler_cache_s3_folder}/rank_{self.rank}_batches.pkl"

        try:
            s3_client = boto3.client("s3")
            self.batches = pickle.loads(s3_client.get_object(Bucket=sampler_cache_s3_bucket, Key=cache_file_name)['Body'].read())
            logger.warning(f"Load self.batches from s3://{sampler_cache_s3_bucket}/{cache_file_name}")
            
        except:
            logger.warning(f"Not found s3://{sampler_cache_s3_bucket}/{cache_file_name}")
            logger.warning("Generate batches")
            self._generate_batches()
            # cache batches to save time for next epoch
            s3_resource = boto3.resource("s3")
            pickle_bytes = pickle.dumps(self.batches)
            s3_resource.Object(sampler_cache_s3_bucket, cache_file_name).put(Body=pickle_bytes)
            logger.warning(f"Dump self.batches to s3://{sampler_cache_s3_bucket}/{cache_file_name}")

    def _generate_batches(self):
        # try:
        #     # If an execution using an identical dataset (order of the datapoints in the dataset must be identical)
        #     # has been triggered before, change length_buckets_file_name to the file saved from the previous execution.
        #     # This saves a lot of time when a corpus is not in memory (e.g., ICQ_Augmented).
        #     s3_client = boto3.client("s3")
        #     length_buckets_file_name = f"flyte/   /length_buckets.pkl"
        #     length_buckets = pickle.loads(s3_client.get_object(Bucket=self.sampler_cache_s3_bucket, Key=length_buckets_file_name)['Body'].read())
        #     logger.warning(f"Load length_buckets from s3://{self.sampler_cache_s3_bucket}/{length_buckets_file_name}")
        #     # check if length_buckets matches dataset (high probability but not guaranteed)
        #     for length, indices in length_buckets.items():
        #         if len(indices) == 1:
        #             continue
        #         index_1, index_2 = random.sample(indices, 2)
        #         if len(self.dataset[index_1]) != len(self.dataset[index_2]):
        #             raise Exception("Incorrect length_buckets_file_name loaded")

        # except:
        logger.warning("Generate length_buckets")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', add_prefix_space=True)
        length_buckets_list = []
        for task_index in range(self.num_tasks):
            if task_index == 0:
                start_idx = 0
            else:
                start_idx = self.cummulative_sizes[task_index-1]
            end_idx = self.cummulative_sizes[task_index]
            length_buckets = defaultdict(list)
            for idx in range(start_idx, end_idx):
                length = len(tokenizer([token.text for token in self.dataset[idx]], is_split_into_words=True)['input_ids'])
                length_buckets[length].append(idx)
            length_buckets_list.append(length_buckets)

        s3_resource = boto3.resource("s3")
        pickle_bytes = pickle.dumps(length_buckets_list)
        s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/length_buckets_list.pkl").put(Body=pickle_bytes)
        logger.warning(f"Dump length_buckets_list to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/length_buckets_list.pkl")

        self.batches = []
        for length_buckets in length_buckets_list:
            sorted_lengths = sorted(length_buckets.keys())
            for length in sorted_lengths:
                num_sentences_per_batch_step = min(self.max_tokens_per_batch_step // length, self.max_sentences_per_batch_step)
                num_sentences_per_batch_step = max(num_sentences_per_batch_step, self.min_sentences_per_batch_step)

                indices_all_GPU = length_buckets[length]
                num_sentences_all_GPU = len(indices_all_GPU)
                num_sentences_per_GPU = num_sentences_all_GPU // self.num_replicas # drop last few sentences
                indices_this_GPU = indices_all_GPU[self.rank : num_sentences_per_GPU * self.num_replicas : self.num_replicas]

                # num_sentences_per_batch = num_sentences_per_batch_step * NUM_STEPS_PER_BATCH
                for i in range(0, num_sentences_per_GPU, num_sentences_per_batch_step):
                    batch_indices = indices_this_GPU[i:i + num_sentences_per_batch_step]
                    if self.skip_ratio is not None:
                        threshold = num_sentences_per_batch_step * self.skip_ratio
                        if len(batch_indices) < threshold:
                            # logger.warning(f"{len(batch_indices)} sentences of length {length} are skipped")
                            break
                    self.batches.append(batch_indices)

        num_indices = sum(len(batch) for batch in self.batches)
        logger.warning(f"num datapoint: {num_indices}")

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            logger.warning(f"shuffle begin from rank {self.rank}")
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(len(self.batches), generator=g).tolist()
            # logger.warning(f"batch_indices of rank {self.rank}: {str(batch_indices)}")
            shuffle_batches = [self.batches[batch_index] for batch_index in batch_indices]
            logger.warning(f"shuffle end from rank {self.rank}")
            return iter(shuffle_batches)
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class SingleLengthSingleTaskAdaptiveBatchSampler(BatchSampler):
    def __init__(
        self, 
        max_tokens_per_batch_step: int = 4096,
        max_sentences_per_batch_step: int = 128,
        min_sentences_per_batch_step: int = 2,
        skip_ratio: Optional[float] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.max_tokens_per_batch_step = max_tokens_per_batch_step
        self.max_sentences_per_batch_step = max_sentences_per_batch_step
        self.min_sentences_per_batch_step = min_sentences_per_batch_step
        self.skip_ratio = skip_ratio
        logger.warning(f"max_tokens_per_batch_step: {max_tokens_per_batch_step}")
        logger.warning(f"max_sentences_per_batch_step: {max_sentences_per_batch_step}")
        logger.warning(f"min_sentences_per_batch_step: {min_sentences_per_batch_step}")
        logger.warning(f"skip_ratio: {skip_ratio}")
        self.shuffle = shuffle
        self.seed = seed
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0
        self.batches = []

    def set_dataset(self, dataset, prev_dev_micro_f1, sampler_cache_s3_bucket, sampler_cache_s3_folder):
        self.dataset = dataset
        self.cummulative_sizes = dataset.cummulative_sizes
        logger.warning(f"cummulative_sizes: {str(dataset.cummulative_sizes)}")
        self.task_ids = dataset.ids
        logger.warning(f"task_ids: {str(dataset.ids)}")
        assert(len(self.cummulative_sizes) == len(self.task_ids))
        self.sampler_cache_s3_bucket = sampler_cache_s3_bucket
        self.sampler_cache_s3_folder = sampler_cache_s3_folder

        length_buckets_list = self._generate_length_buckets()

        self.batches = []
        for task_index, length_buckets in enumerate(length_buckets_list):
            task_id = self.task_ids[task_index]
            sample_ratio = 1 - (prev_dev_micro_f1[task_id]) ** 2
            logger.warning(f"For {task_id}, sample_ratio is {sample_ratio}")

            sorted_lengths = sorted(length_buckets.keys())
            for length in sorted_lengths:
                num_sentences_per_batch_step = min(self.max_tokens_per_batch_step // length, self.max_sentences_per_batch_step)
                num_sentences_per_batch_step = max(num_sentences_per_batch_step, self.min_sentences_per_batch_step)

                indices_all_GPU = length_buckets[length]
                num_sentences_all_GPU = len(indices_all_GPU)
                num_sentences_per_GPU = num_sentences_all_GPU // self.num_replicas # drop last few sentences
                indices_this_GPU = indices_all_GPU[self.rank : num_sentences_per_GPU * self.num_replicas : self.num_replicas]

                sample_size = int(sample_ratio * len(indices_this_GPU))
                if sample_size < len(indices_this_GPU):
                    indices_this_GPU = random.sample(indices_this_GPU, sample_size)

                # num_sentences_per_batch = num_sentences_per_batch_step * NUM_STEPS_PER_BATCH
                for i in range(0, sample_size, num_sentences_per_batch_step):
                    batch_indices = indices_this_GPU[i:i + num_sentences_per_batch_step]
                    if self.skip_ratio is not None:
                        threshold = num_sentences_per_batch_step * self.skip_ratio
                        if len(batch_indices) < threshold:
                            # logger.warning(f"{len(batch_indices)} sentences of length {length} are skipped")
                            break
                    self.batches.append(batch_indices)

        num_indices = sum(len(batch) for batch in self.batches)
        logger.warning(f"num datapoint: {num_indices}")


    def _generate_length_buckets(self):
        try:
            s3_client = boto3.client("s3")
            length_buckets_list_file_path = f"{self.sampler_cache_s3_folder}/length_buckets_list.pkl"
            length_buckets_list = pickle.loads(s3_client.get_object(Bucket=self.sampler_cache_s3_bucket, Key=length_buckets_list_file_path)['Body'].read())
            logger.warning(f"Load length_buckets_list from s3://{self.sampler_cache_s3_bucket}/{length_buckets_list_file_path}")
            # loading from same execution, no need to check
            # # check if length_buckets matches dataset (high probability but not guaranteed)
            # for length, indices in length_buckets.items():
            #     if len(indices) == 1:
            #         continue
            #     index_1, index_2 = random.sample(indices, 2)
            #     if len(self.dataset[index_1]) != len(self.dataset[index_2]):
            #         raise Exception("Incorrect length_buckets_file_name loaded")

        except:
            logger.warning("Generate length_buckets_list")
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', add_prefix_space=True)
            length_buckets_list = []
            for task_index in range(len(self.task_ids)):
                if task_index == 0:
                    start_idx = 0
                else:
                    start_idx = self.cummulative_sizes[task_index-1]
                end_idx = self.cummulative_sizes[task_index]
                length_buckets = defaultdict(list)
                for idx in range(start_idx, end_idx):
                    length = len(tokenizer([token.text for token in self.dataset[idx]], is_split_into_words=True)['input_ids'])
                    length_buckets[length].append(idx)
                length_buckets_list.append(length_buckets)

            if self.rank == 0:
                s3_resource = boto3.resource("s3")
                pickle_bytes = pickle.dumps(length_buckets_list)
                s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/length_buckets_list.pkl").put(Body=pickle_bytes)
                logger.warning(f"Dump length_buckets_list to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/length_buckets_list.pkl")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return length_buckets_list



    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            logger.warning(f"shuffle begin from rank {self.rank}")
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(len(self.batches), generator=g).tolist()
            # logger.warning(f"batch_indices of rank {self.rank}: {str(batch_indices)}")
            shuffle_batches = [self.batches[batch_index] for batch_index in batch_indices]
            logger.warning(f"shuffle end from rank {self.rank}")
            return iter(shuffle_batches)
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class SingleTaskBatchSampler(BatchSampler):
    def __init__(
        self, 
        mini_batch_chunk_size: int = 2,
        shuffle: bool = True,
        seed: int = 0,
    ):
        # drop_last = True
        self.mini_batch_chunk_size = mini_batch_chunk_size
        logger.warning(f"mini_batch_chunk_size: {mini_batch_chunk_size}")
        self.shuffle = shuffle
        self.seed = seed
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0
        # indices of datapoints from different tasks are in different lists
        self.indices: list[list[int]] = []


    def set_dataset(self, dataset, sampler_cache_s3_bucket, sampler_cache_s3_folder):
        self.dataset = dataset
        self.sampler_cache_s3_bucket = sampler_cache_s3_bucket
        self.sampler_cache_s3_folder = sampler_cache_s3_folder
        self.cummulative_sizes = dataset.cummulative_sizes
        logger.warning(f"cummulative_sizes: {str(dataset.cummulative_sizes)}")
        self.num_tasks = len(self.cummulative_sizes)

        for task_index in range(self.num_tasks):
            if task_index == 0:
                start_idx = 0
            else:
                start_idx = self.cummulative_sizes[task_index-1]
            end_idx = self.cummulative_sizes[task_index]
            num_datapoints = end_idx - start_idx
            num_datapoints_per_GPU = num_datapoints // self.num_replicas # drop
            num_datapoints_per_GPU = num_datapoints_per_GPU // self.mini_batch_chunk_size * self.mini_batch_chunk_size # drop
            indices_one_task_this_GPU = list(range(start_idx+self.rank, start_idx+num_datapoints_per_GPU*self.num_replicas, self.num_replicas))
            if indices_one_task_this_GPU:
                self.indices.append(indices_one_task_this_GPU)

    def __iter__(self):
        if not self.shuffle:
            batches = []
            for indices_one_task in self.indices:
                for i in range(0, len(indices_one_task), self.mini_batch_chunk_size):
                    batches.append(indices_one_task[i: i+self.mini_batch_chunk_size])
            
            s3_resource = boto3.resource("s3")
            pickle_bytes = pickle.dumps(batches)
            s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/batches_rank_{self.rank}.pkl").put(Body=pickle_bytes)
            logger.warning(f"Dump batches to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/batches_rank_{self.rank}.pkl")
            return iter(batches)


        # deterministically shuffle based on epoch, seed and rank
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank)
        shuffled_indices = []
        for indices_one_task in self.indices:
            order = torch.randperm(len(indices_one_task), generator=g).tolist()
            shuffled_indices.append([indices_one_task[i] for i in order])
        
        batches = []
        for indices_one_task in shuffled_indices:
            for i in range(0, len(indices_one_task), self.mini_batch_chunk_size):
                batches.append(indices_one_task[i: i+self.mini_batch_chunk_size])

        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=g).tolist()
        shuffled_batches = [batches[i] for i in order]
        
        s3_resource = boto3.resource("s3")
        pickle_bytes = pickle.dumps(shuffled_batches)
        # s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/shuffled_batches_rank_{self.rank}.pkl").put(Body=pickle_bytes)
        # logger.warning(f"Dump batches to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/shuffled_batches_rank_{self.rank}.pkl")

        return iter(shuffled_batches)

    

    def __len__(self):
        return sum(len(indices_one_task) // self.mini_batch_chunk_size for indices_one_task in self.indices)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class SingleTaskAdaptiveBatchSampler(BatchSampler):
    def __init__(
        self, 
        mini_batch_chunk_size: int = 2,
        shuffle: bool = True,
        seed: int = 0,
    ):
        # drop_last = True
        self.mini_batch_chunk_size = mini_batch_chunk_size
        logger.warning(f"mini_batch_chunk_size: {mini_batch_chunk_size}")
        self.shuffle = shuffle
        self.seed = seed
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.epoch = 0
        # indices of datapoints from different tasks are in different lists
        self.indices: list[list[int]] = []


    def set_dataset(self, dataset, prev_dev_micro_f1, sampler_cache_s3_bucket, sampler_cache_s3_folder):
        self.dataset = dataset
        self.sampler_cache_s3_bucket = sampler_cache_s3_bucket
        self.sampler_cache_s3_folder = sampler_cache_s3_folder
        self.cummulative_sizes = dataset.cummulative_sizes
        logger.warning(f"cummulative_sizes: {str(dataset.cummulative_sizes)}")
        self.task_ids = dataset.ids
        logger.warning(f"task_ids: {str(dataset.ids)}")
        assert(len(self.cummulative_sizes) == len(self.task_ids))

        for task_index, task_id in enumerate(self.task_ids):
            if task_index == 0:
                start_idx = 0
            else:
                start_idx = self.cummulative_sizes[task_index-1]
            end_idx = self.cummulative_sizes[task_index]
            num_datapoints = end_idx - start_idx
            num_datapoints_per_GPU = num_datapoints // self.num_replicas # drop
            num_datapoints_per_GPU = num_datapoints_per_GPU // self.mini_batch_chunk_size * self.mini_batch_chunk_size # drop
            indices_one_task_this_GPU = list(range(start_idx+self.rank, start_idx+num_datapoints_per_GPU*self.num_replicas, self.num_replicas))
            sample_ratio = 1 - (prev_dev_micro_f1[task_id]) ** 2
            sample_size = int(sample_ratio * len(indices_one_task_this_GPU))
            logger.warning(f"For {task_id}, sample_ratio is {sample_ratio} and sample_size is {sample_size}")
            if sample_size < len(indices_one_task_this_GPU):
                indices_one_task_this_GPU = random.sample(indices_one_task_this_GPU, sample_size)
            if indices_one_task_this_GPU:
                self.indices.append(indices_one_task_this_GPU)

    def __iter__(self):
        if not self.shuffle:
            batches = []
            for indices_one_task in self.indices:
                for i in range(0, len(indices_one_task), self.mini_batch_chunk_size):
                    batches.append(indices_one_task[i: i+self.mini_batch_chunk_size])
            
            s3_resource = boto3.resource("s3")
            pickle_bytes = pickle.dumps(batches)
            s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/batches_rank_{self.rank}.pkl").put(Body=pickle_bytes)
            logger.warning(f"Dump batches to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/batches_rank_{self.rank}.pkl")
            return iter(batches)


        # deterministically shuffle based on epoch, seed and rank
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank)
        shuffled_indices = []
        for indices_one_task in self.indices:
            order = torch.randperm(len(indices_one_task), generator=g).tolist()
            shuffled_indices.append([indices_one_task[i] for i in order])
        
        batches = []
        for indices_one_task in shuffled_indices:
            for i in range(0, len(indices_one_task), self.mini_batch_chunk_size):
                batches.append(indices_one_task[i: i+self.mini_batch_chunk_size])

        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=g).tolist()
        shuffled_batches = [batches[i] for i in order]
        
        s3_resource = boto3.resource("s3")
        pickle_bytes = pickle.dumps(shuffled_batches)
        # s3_resource.Object(self.sampler_cache_s3_bucket, f"{self.sampler_cache_s3_folder}/shuffled_batches_rank_{self.rank}.pkl").put(Body=pickle_bytes)
        # logger.warning(f"Dump batches to s3://{self.sampler_cache_s3_bucket}/{self.sampler_cache_s3_folder}/shuffled_batches_rank_{self.rank}.pkl")

        return iter(shuffled_batches)

    

    def __len__(self):
        return sum(len(indices_one_task) // self.mini_batch_chunk_size for indices_one_task in self.indices)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


