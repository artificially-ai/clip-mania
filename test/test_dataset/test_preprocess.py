import os

from unittest import TestCase

from pathlib import Path

from clip_mania.utils.data.preprocess import DatasetProcessor


class TestDatasetProcessor(TestCase):

    def setUp(self):
        self.current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def test_create_dataset(self):
        dataset_path = os.path.join(self.current_path, 'dataset/train')
        dataset = DatasetProcessor.create_dataset(dataset_path)
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) == 1)
        self.assertTrue(len(dataset['items']) == 6)
