import queue as Queue
import threading
from typing import Iterable
import numpy as np
import torch
from PIL import Image
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
# Custom modules
from .datasets import MIMICA # to open X_A.jpg
from .utils_distributed_sampler import DistributedSampler
from .utils_distributed_sampler import get_dist_info, worker_init_fn

def get_dataloader(args, local_rank) -> Iterable:
    rank, world_size = get_dist_info()

    trainSet = MIMICA(args.csv_path, 
                    args.image_root,
                    mode='train',
                    input_nc=args.input_nc,
                    use_transform = args.use_transform,
                    normalize = args.normalize,
                    shuffle=True,
                    oui = args.oui)
    
    train_sampler = DistributedSampler(trainSet, num_replicas=world_size, rank=rank, 
                                       shuffle=True, seed=args.seed)
    if args.seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=args.num_workers, rank=rank, seed=args.seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainSet,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader, trainSet.__len__()

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

