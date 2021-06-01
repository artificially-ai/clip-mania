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

    def test_create_indexed_labels(self):
        dataset_path = os.path.join(self.current_path, 'dataset/train')
        prompts = DatasetProcessor.create_indexed_prompts(dataset_path)
        self.assertIsNotNone(prompts)
        self.assertTrue(len(prompts) == 2)
        labels = [0, 1]
        expected_message = "This is a picture of a(n) boat."
        self.assertTrue(expected_message in prompts)
        self.assertTrue(prompts[expected_message] in labels)

        given_key = [expected_message]
        extracted_labels = [label for key, label in prompts.items() if key in given_key]

        self.assertTrue(extracted_labels == [0])
