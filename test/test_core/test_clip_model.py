import os

from unittest import TestCase

from pathlib import Path

import numpy as np

import torch
import clip

import PIL


class TestModelLoading(TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def test_load_model(self):
        self.assertIsNotNone(self.model)

    def test_inference(self):
        classes = ["a diagram", "a dog", "a cat", "an airplane"]
        image_path = os.path.join(self.current_path, "dataset/train/airplane/airplane1.jpg")
        image = self.preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(self.device)
        text = clip.tokenize(classes).to(self.device)

        with torch.no_grad():
            _image_features = self.model.encode_image(image)
            _text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_index = np.argmax(probs)
        prediction = classes[max_index]
        prob = probs.flatten()[max_index]
        self.assertTrue(prob > 0.9)
        self.assertTrue(prediction == "an airplane")
