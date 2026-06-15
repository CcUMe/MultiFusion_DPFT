from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset, Subset, default_collate

try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

from dprt.utils.misc import as_list


def listed_collating(
        data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Attributes:
        data: List to data tuples consisting of input and target values.

    Returns:
        batch: Batched data consisting of a tuple of batched inputs and
            a list of targets.
    """
    # Split data into inputs and targets (list of tuples to tuple of lists)
    inputs, targets = zip(*data)

    # Ensure list data type
    inputs = as_list(inputs)
    targets = as_list(targets)

    # Convert tensors to batch of tensors
    inputs = default_collate(inputs)

    # Combine inputs and outputs
    batch = (inputs, targets)

    return batch


def apply_subset(dataset: Dataset, config: Dict[str, Any]) -> Dataset:
    data_config = config.get('data', {})
    split = getattr(dataset, 'split', None)
    subset_size = data_config.get(f'{split}_subset') if split else None
    if subset_size is None:
        subset_size = data_config.get('subset')
    if subset_size in (None, False):
        return dataset

    subset_size = min(int(subset_size), len(dataset))
    return Subset(dataset, range(subset_size))


def load_listed(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
    dataset = apply_subset(dataset, config)
    split = getattr(dataset, 'split', None)
    shuffle = bool(config['train']['shuffle']) if split in {None, 'train'} else False
    num_workers = int(config['computing']['workers'])
    return DataLoader(
        dataset=dataset,
        batch_size=config['train']['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=listed_collating
    )
