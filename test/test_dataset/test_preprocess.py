from unittest import TestCase

from clip_mania.utils.data.preprocess import DatasetProcessor


class TestDatasetProcessor(TestCase):

    def test_create_dataset(self):
        dataset = DatasetProcessor.create_dataset('./dataset/train')
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) == 1)
