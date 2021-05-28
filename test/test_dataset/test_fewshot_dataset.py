import os

from unittest import TestCase

from pathlib import Path

from clip_mania.utils.data.datasets import FewShotDataset


class TestFewShotDataSet(TestCase):

    def setUp(self):
        self.current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def test_empty_dataset(self):
        dataset_path = os.path.join(self.current_path, 'dataset/test')
        dataset = FewShotDataset(dataset_path)
        n_samples = len(dataset)
        self.assertTrue(n_samples == 0)

    def test_load_dataset(self):
        dataset_path = os.path.join(self.current_path, 'dataset/train')
        dataset = FewShotDataset(dataset_path)
        n_samples = len(dataset)
        self.assertTrue(n_samples == 6)

        for i in range(n_samples):
            image, prompt = dataset[i]
            self.assertIsNotNone(image)
            self.assertIsNotNone(prompt)
