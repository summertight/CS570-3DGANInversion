import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import pickle
import glob
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from PIL import Image

'''
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
'''

def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    # if root_dir == "synthetic":
    #     train_set = SyntheticDataset()
    #     dali = False
    # # Mxnet RecordIO
    # elif os.path.exists(rec) and os.path.exists(idx):
    #     train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Image Folder
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    # train_set = ImageFolder(root_dir, transform)
    train_set = WebFace42M(root_dir, transform)

    # DALI
    # if dali:
    #     return dali_data_iter(
    #         batch_size=batch_size, rec_file=rec, idx_file=idx,
    #         num_threads=2, local_rank=local_rank)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

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


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


class WebFace42M(Dataset):
    '''
    WebFace42M dataset
    webface_root = '/home/nas1_userB/dataset/WebFace42M/img_folder'
    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms):
        self.root_dir = imgs_folder
        self.transform = train_transforms

        with open('/home/nas4_user/jungsoolee/Face_dataset/webface_str2label.pickle', 'rb') as f:
            self.str2label = pickle.load(f)

        # CLASS LIST
        if os.path.exists(os.path.join(imgs_folder, 'webface_class_list.pkl')):
            with open(os.path.join(imgs_folder, 'webface_class_list.pkl'), 'rb') as f:
                self.class_num = len(pickle.load(f))
        else:
            class_list = os.listdir(imgs_folder)
            self.class_num = len(os.listdir(imgs_folder))
            with open(os.path.join(imgs_folder, 'webface_class_list.pkl'), 'wb') as f:
                pickle.dump(class_list, f)

        # TOTAL LIST
        if os.path.exists(os.path.join(imgs_folder, 'webface_total_list.pkl')):
            with open(os.path.join(imgs_folder, 'webface_total_list.pkl'), 'rb') as f:
                total_list = pickle.load(f)
        else:
            total_list = glob.glob(self.root_dir + '/*/*')
            with open(os.path.join(imgs_folder, 'webface_total_list.pkl'), 'wb') as f:
                pickle.dump(total_list, f)

        self.total_imgs = len(total_list)

        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')
    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]

        img = Image.open(img_path)
        # label = int(img_path.split('/')[-2].split('_')[-1] + img_path.split()) # CHECK
        label = self.str2label[img_path.split('/')[-2]]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
