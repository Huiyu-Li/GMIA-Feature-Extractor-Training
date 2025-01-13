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
from .datasets import MIMIC
from .utils_distributed_sampler import DistributedSampler
from .utils_distributed_sampler import get_dist_info, worker_init_fn

def get_dataloader(args, local_rank) -> Iterable:
    rank, world_size = get_dist_info()

    trainSet = MIMIC(args.csv_path, 
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

class MIMIC_EVAL(Dataset):
    r"""
        Iterating: two real image pairs
        For CallBackVerification during training
    """
    def __init__(self, 
                 csv_path,
                 image_root='',
                 input_nc = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        
        self.image_root = image_root
        self.input_nc=input_nc
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)
        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

    def __len__(self):
        return self._num_pairs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id_1"])[:2]}/p{str(row["subject_id_1"])}/s{str(row["study_id_1"])}/{row["dicom_id_1"]}.jpg'
        fname2 = f'p{str(row["subject_id_2"])[:2]}/p{str(row["subject_id_2"])}/s{str(row["study_id_2"])}/{row["dicom_id_2"]}.jpg'
        issame = row["issame"]
        image_path1 = f'{self.image_root}/{fname1}'
        image_path2 = f'{self.image_root}/{fname2}'

        if self.input_nc==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame

class MIMICI(Dataset):
    r"""
         Iterating: two anonymized image pairs
         For Linkability Inner Risk computing: Whether or not two anonymized image belong to the same patient
    """
    def __init__(self, 
                 csv_path,
                 image_root='',
                 input_nc = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        
        self.image_root = image_root
        self.input_nc=input_nc
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)
        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

    def __len__(self):
        return self._num_pairs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id_1"])[:2]}/p{str(row["subject_id_1"])}/s{str(row["study_id_1"])}/{row["dicom_id_1"]}/X_A.jpg'
        fname2 = f'p{str(row["subject_id_2"])[:2]}/p{str(row["subject_id_2"])}/s{str(row["study_id_2"])}/{row["dicom_id_2"]}/X_A.jpg'
        issame = row["issame"]
        image_path1 = f'{self.image_root}/{fname1}'
        image_path2 = f'{self.image_root}/{fname2}'

        if self.input_nc==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame

class MIMICO(Dataset):
    r"""
        Iterating: real-anonymized image pairs 
        For Linkability Outer Risk computing: Whether or not a real image and its anonymized one belong to the same patient
    """
    def __init__(self, 
                 csv_path,
                 image_root='',
                 anony_root='',
                 mode = 'test',
                 input_nc = 1,
                 transform = [False, False], # [RandomHorizontalFlip, Rotation]
                 shuffle=False,
                 seed=123,
                 ):
        
        self.image_root = image_root
        self.anony_root = anony_root
        self.mode = mode
        self.input_nc=input_nc
        self.transform = transform

        # load data from csv
        self.df = pd.read_csv(csv_path)
        
        # read train-valid-test split
        if self.mode == "train":
            self.df = self.df[self.df['split']=='train']
        elif self.mode == "valid":
            self.df = self.df[self.df['split']=='validate']
        elif self.mode == "test":
            self.df = self.df[self.df['split']=='test']
        else:
            raise NotImplementedError(f"split {self.mode} is not implemented!")

        # Test
        # self.df = self.df[:1000]

        self._num_pairs = len(self.df)

        # shuffle data
        if shuffle:
            data_index = list(range(self._num_pairs))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

    def __len__(self):
        return self._num_pairs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname1 = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}.jpg'
        fname2 = f'p{str(row["subject_id"])[:2]}/p{str(row["subject_id"])}/s{str(row["study_id"])}/{row["dicom_id"]}/X_A.jpg'
        issame = 1 # 1:if only use same pairs
        image_path1 = f'{self.image_root}/{fname1}'
        image_path2 = f'{self.anony_root}/{fname2}'

        if self.input_nc==3:
            image1 = Image.open(image_path1).convert('RGB') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('RGB') # PIL.Image.Image, [Columns, Rows]
        else:
            image1 = Image.open(image_path1).convert('L') # PIL.Image.Image, [Columns, Rows]
            image2 = Image.open(image_path2).convert('L') # PIL.Image.Image, [Columns, Rows]

        if all(self.transform):
            image1_hflip = F.hflip(image1)
            image2_hflip = F.hflip(image2)
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip), F.to_tensor(image2_rotate)]
        elif self.transform[0]:
           image1_hflip = F.hflip(image1)
           image2_hflip = F.hflip(image2)
           image1 = [F.to_tensor(image1), F.to_tensor(image1_hflip)] 
           image2 = [F.to_tensor(image2), F.to_tensor(image2_hflip)]
        elif self.transform[1]:
            image1_rotate = F.rotate(image1, angle=15)
            image2_rotate = F.rotate(image2, angle=15)
            image1 = [F.to_tensor(image1), F.to_tensor(image1_rotate)] 
            image2 = [F.to_tensor(image2), F.to_tensor(image2_rotate)]
        else:
            image1 = F.to_tensor(image1)
            image2 = F.to_tensor(image2)
        return image1, image2, issame