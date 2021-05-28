from typing import List, Tuple

import clip

import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from clip_mania.utils.data.datasets import FewShotDataset


class ModelExecutor:

    def __init__(self, batch_size=2, lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2):
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Helper method to abstract a direct call to CLIP.

        :return: List
            list of available CLIP models.
        """
        return clip.available_models()

    @staticmethod
    def convert_models_to_fp32(clip_model) -> None:
        """
        This method converts the weights and gradients to fp32 before the optmisation step takes place. After that,
        the precision is halved to speed up the matrix multiplication during batch normalisation.

        Source: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
        """
        for p in clip_model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    def train(self, dataset_path, model_name="ViT-B/32", epochs=10) -> Tuple[nn.Module, Compose]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # No need to pass the device as the clip.load call already has it.
        clip_model, preprocess = clip.load(name=model_name, jit=False)

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = Adam(clip_model.parameters(), lr=self.lr, betas=self.betas,
                         eps=self.eps, weight_decay=self.weight_decay)

        dataset = FewShotDataset(dataset_path)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size)
        for epoch in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()

                images, prompts = batch

                images = torch.stack([img for img in images], dim=0)
                prompts = clip.tokenize(prompts)

                logits_per_image, logits_per_text = clip_model(images, prompts)
                if device == "cpu":
                    ground_truth = torch.arange(self.batch_size).long().to(device)
                else:
                    ground_truth = torch.arange(self.batch_size).half().to(device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss.backward()
                if device == "cpu":
                    optimizer.step()
                else:
                    self.convert_models_to_fp32(clip_model)
                    optimizer.step()
                    clip_model.model.convert_weights(clip_model)
        return clip_model, preprocess
