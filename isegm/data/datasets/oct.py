import cv2
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import os

class OCTDataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(OCTDataset, self).__init__(**kwargs)
        assert split in {'train', 'val', 'test'}
        self.name = 'OCT'
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._gt_path = self.dataset_path / 'gt'
        self._list_path = self.dataset_path / 'list'

        with open(self._list_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.png')
        gt_path = str(self._gt_path / f'{image_name}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        gt = cv2.resize(gt, (256, 256))
        gt = (gt > 128).astype(np.int32)

        return DSample(image, gt, objects_ids=[1], sample_id=index)