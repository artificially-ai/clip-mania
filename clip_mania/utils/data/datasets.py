import torch

from torch.utils.data import Dataset
from torchvision import transforms as T

import PIL

import pandas as pd

from clip_mania.utils.data.preprocess import DatasetProcessor


class FewShotDataset(Dataset):

    def __init__(self, root_dir, image_size=224, resize_ratio=0.75, shuffle=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        raw_dataset = DatasetProcessor.create_dataset(root_dir)
        self.df_dataset = pd.DataFrame(raw_dataset)

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_file = self.df_dataset.iloc[item][0][0]
        image_tensor = self.transform(PIL.Image.open(img_file))
        prompt = self.df_dataset.iloc[item][0][1]

        return image_tensor, prompt
