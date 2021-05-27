from unittest import TestCase

import torch
import clip


class TestModelLoading(TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def test_load_model(self):
        self.assertIsNotNone(self.model)
