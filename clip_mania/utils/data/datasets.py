import io

import torch
from torch.utils.data import Dataset

from skimage import io

import pandas as pd

from clip_mania.utils.data.preprocess import DatasetProcessor


class FewShotDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        raw_dataset = DatasetProcessor.create_dataset(root_dir)
        self.df_dataset = pd.DataFrame(raw_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_name = self.df_dataset.iloc[item][0][0]
        image = io.imread(img_name)
        prompt = self.df_dataset.iloc[item][0][1]

        if self.transform:
            image = self.transform(image)

        return image, prompt
