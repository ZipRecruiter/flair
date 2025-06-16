from collections import defaultdict

from flair.data import ConcatFlairDataset, Sentence
from flair.samplers import (
    SingleTaskAdaptiveBatchSampler,
    SingleTaskSingleLengthAdaptiveBatchSampler,
)


def test_single_task_adaptive_batch_sampler():
    task_ids = ["task_1", "task_2", "task_3"]
    dataset_1 = [Sentence(f"sentence {i} in {task_ids[0]}") for i in range(16)]  # 16 sentences
    dataset_2 = [Sentence(f"sentence {i} in {task_ids[1]}") for i in range(25)]  # 25 sentences
    dataset_3 = [Sentence(f"sentence {i} in {task_ids[2]}") for i in range(34)]  # 34 sentences
    concat_dataset = ConcatFlairDataset([dataset_1, dataset_2, dataset_3], ids=task_ids)

    sampler = SingleTaskAdaptiveBatchSampler(dataset=concat_dataset, batch_size=4, seed=0)
    sampler.generate_batches(
        prev_dev_micro_f1={}, shuffle=True, epoch=1
    )  # prev_dev_micro_f1 is not used because downsample_ratio_func is None
    generated_batches = iter(sampler)
    visited_index = set()
    for batch in generated_batches:
        task_id_first_sentence = concat_dataset[batch[0]].get_label("multitask_id").value
        # ensure each index is unique
        assert batch[0] not in visited_index
        visited_index.add(batch[0])
        for datapoint_idx in batch[1:]:
            # ensure each batch contains data points from the same task
            assert concat_dataset[datapoint_idx].get_label("multitask_id").value == task_id_first_sentence
            # ensure each index is unique
            assert datapoint_idx not in visited_index
            visited_index.add(datapoint_idx)
    # ensure all indices have been visited
    assert len(visited_index) == len(concat_dataset)


def test_single_task_single_length_adaptive_batch_sampler():
    task_ids = ["task_1", "task_2", "task_3"]
    dataset_1 = [Sentence(f"sentence {i} in {task_ids[0]}") for i in range(8)]  # 8 sentences of length 6
    dataset_1 += [
        Sentence(f"sentence {i} in {task_ids[0]} (this is a {'long ' * 10}sentence)") for i in range(8, 16)
    ]  # 8 sentences of length 22
    dataset_2 = [Sentence(f"sentence {i} in {task_ids[1]}") for i in range(12)]  # 12 sentences of length 6
    dataset_2 += [
        Sentence(f"sentence {i} in {task_ids[1]} (this is a {'long ' * 10}sentence)") for i in range(12, 25)
    ]  # 13 sentences of length 22
    dataset_3 = [Sentence(f"sentence {i} in {task_ids[2]}") for i in range(17)]  # 17 sentences of length 6
    dataset_3 += [
        Sentence(f"sentence {i} in {task_ids[2]} (this is a {'long ' * 10}sentence)") for i in range(17, 34)
    ]  # 17 sentences of length 22
    concat_dataset = ConcatFlairDataset([dataset_1, dataset_2, dataset_3], ids=task_ids)

    def create_length_dict(dataset: list[Sentence]) -> dict[int, list[int]]:
        length_to_indices = defaultdict(list)
        for idx, sent in enumerate(dataset):
            length_to_indices[len(sent)].append(idx)
        return length_to_indices

    length_dicts_for_tasks = {
        task_ids[0]: create_length_dict(dataset_1),
        task_ids[1]: create_length_dict(dataset_2),
        task_ids[2]: create_length_dict(dataset_3),
    }

    def batch_size_func(length: int) -> int:
        if length <= 10:
            return 4
        return 2

    sampler = SingleTaskSingleLengthAdaptiveBatchSampler(
        dataset=concat_dataset,
        length_dicts_for_tasks=length_dicts_for_tasks,
        batch_size_func=batch_size_func,
        min_batch_size=2,  # The smallest integer that batch_size_func could return is 2
        skip_ratio=1,  # skip underfilled batches
        seed=0,
    )
    sampler.generate_batches(
        prev_dev_micro_f1={}, shuffle=True, epoch=1
    )  # prev_dev_micro_f1 is not used because downsample_ratio_func is None
    generated_batches = iter(sampler)
    visited_index = set()
    for batch in generated_batches:
        task_id_first_sentence = concat_dataset[batch[0]].get_label("multitask_id").value
        length_first_sentence = len(concat_dataset[batch[0]])
        # ensure each index is unique
        assert batch[0] not in visited_index
        # ensure batch size equals expected value
        assert len(batch) == batch_size_func(length_first_sentence)
        visited_index.add(batch[0])
        for datapoint_idx in batch[1:]:
            # ensure each batch contains data points from the same task
            assert concat_dataset[datapoint_idx].get_label("multitask_id").value == task_id_first_sentence
            # ensure each batch contains data points of same length
            assert len(concat_dataset[datapoint_idx]) == length_first_sentence
            # ensure each index is unique
            assert datapoint_idx not in visited_index
            visited_index.add(datapoint_idx)
    # ensure number of batches equals expected value
    assert len(sampler) == 8 // 2 + 8 // 4 + 12 // 2 + 13 // 4 + 17 // 2 + 17 // 4
