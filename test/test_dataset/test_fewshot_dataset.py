from unittest import TestCase

from clip_mania.utils.data.datasets import FewShotDataset


class TestFewShotDataSet(TestCase):

    def test_empty_dataset(self):
        dataset = FewShotDataset('./dataset/test')
        n_samples = len(dataset)
        self.assertTrue(n_samples == 0)

    def test_load_dataset(self):
        dataset = FewShotDataset('./dataset/train')
        n_samples = len(dataset)
        self.assertTrue(n_samples == 6)

        for i in range(n_samples):
            image, prompt = dataset[i]
            self.assertIsNotNone(image)
            self.assertIsNotNone(prompt)
