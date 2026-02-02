"""
Dataset loading utilities for CycleGAN.
"""

import importlib
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import and return the dataset class matching `dataset_name`."""
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create and return a dataset given parsed options."""
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Dataset wrapper with optional distributed sampling."""

    def __init__(self, opt):
        """Initialize this class given the options."""
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        if "LOCAL_RANK" in os.environ:
            print(f'create DDP sampler on rank {int(os.environ["LOCAL_RANK"])}')
            self.sampler = DistributedSampler(self.dataset, shuffle=not opt.serial_batches)
            shuffle = False
        else:
            self.sampler = None
            shuffle = not opt.serial_batches

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=shuffle, sampler=self.sampler, num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def set_epoch(self, epoch):
        """Set epoch for DistributedSampler to ensure proper shuffling"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
