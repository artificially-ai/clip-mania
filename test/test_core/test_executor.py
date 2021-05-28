import os

from unittest import TestCase

from pathlib import Path

import PIL

import torch
import clip

import numpy as np

from clip_mania.core.executor import ModelExecutor


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
        executor = ModelExecutor()
        model, preprocess = executor.train(dataset_path, epochs=2)
        self.assertIsNotNone(model)
        self.assertIsNotNone(preprocess)

        classes = ["a bird", "a dog", "a cat", "a airplane"]
        image_path = os.path.join(self.current_path, "dataset/test/airplane/airplane.jpg")

        image = preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(classes).to(self.device)

        with torch.no_grad():
            _image_features = model.encode_image(image)
            _text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_index = np.argmax(probs)
        prediction = classes[max_index]
        expected_prob = probs.flatten()[3]
        highest_prob = probs.flatten()[max_index]
        print(f"Expected 'an airplane', but got '{prediction}'")
        print(f"Probability for the expected prompt was '{expected_prob:.4f}'")
        print(f"Highest probability was '{highest_prob:.4f}'")