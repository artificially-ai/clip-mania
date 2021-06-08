import os

from unittest import TestCase

from pathlib import Path

import PIL

import torch
import clip

import numpy as np

from clip_mania.core.executor import ModelExecutor
from clip_mania.utils.data.preprocess import DatasetProcessor


class TestModelExecutor(TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def test_instance(self):
        executor = ModelExecutor()
        self.assertIsNotNone(executor)

        models = ModelExecutor.get_available_models()
        self.assertIsNotNone(models)
        self.assertTrue("ViT-B/32" in models)

    def test_train(self):
        dataset_path = os.path.join(self.current_path, "dataset/train")
        batch_size = 2  # number of classes
        executor = ModelExecutor(batch_size=batch_size, lr=1e-8, weight_decay=0.1)
        model, preprocess = executor.train(dataset_path, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(preprocess)

        prompts = DatasetProcessor.create_indexed_prompts(dataset_path)
        classes = list(prompts.keys())
        image_path = os.path.join(self.current_path, "dataset/test/airplane/airplane1.jpg")

        image = preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(classes).to(self.device)

        with torch.no_grad():
            _image_features = model.encode_image(image)
            _text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_index = np.argmax(probs)
        prediction = classes[max_index]
        expected_prob = probs.flatten()[0]
        highest_prob = probs.flatten()[max_index]
        self.assertTrue(expected_prob > 0.7)
        self.assertTrue(expected_prob == highest_prob)
        self.assertTrue(prediction == "This is a picture of a(n) airplane.")
        print(f"\nExpected 'This is a picture of a(n) airplane.' and  got '{prediction}'")
        print(f"Probability for the expected prompt was '{expected_prob:.4f}'")
        print(f"Expected probability was '{expected_prob:.4f}'")
        print(f"Highest probability was '{highest_prob:.4f}'")
