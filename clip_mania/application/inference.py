import os
import glob
import re

import PIL
import numpy as np

import pandas as pd

import torch
import clip

from absl import app, flags, logging
from absl.flags import FLAGS

from tqdm import tqdm

from clip_mania.utils.data.preprocess import DatasetProcessor

flags.DEFINE_string(name='test_dataset_path', default=None, help='Absolute path to the test dataset location.',
                    required=True)

flags.DEFINE_string(name='model_path', default=None, help='Absolute path where to load the model from.',
                    required=True)

flags.register_validator('test_dataset_path',
                         lambda value: type(value) is str and os.path.isdir(value),
                         message='--test_dataset_path must be a valid directory.')

flags.register_validator('model_path',
                         lambda value: type(value) is str and value is not None,
                         message='--model_path must be a valid directory.')


def main(_args):
    test_dataset_path = FLAGS.test_dataset_path
    model_path = FLAGS.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Loading the pre-trained model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    logging.info("Loading the custom model's state dictionary...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    indexed_prompts = DatasetProcessor.create_indexed_prompts(test_dataset_path)
    prompts = list(indexed_prompts.keys())
    text = clip.tokenize(prompts).to(device)

    # Please, make sure that you have only images in the inference dataset.
    full_path = os.path.join(test_dataset_path, "**/*.*")
    images = glob.glob(full_path, recursive=True)

    expected_pattern = r"(?<=/test/)((?!=\D)).+(?=/(?=[\d|\D]))"
    regex_e = re.compile(expected_pattern)
    # I like this Regex. :)
    # predicted_pattern = r"(?<=\)\s).*[^\.]"
    # regex_p = re.compile(predicted_pattern)

    results = {"y": [], "y_hat": [], "predicted_prompt": [], "probability": []}
    logging.info("Starting inference...")
    for image_path in tqdm(images):
        image = preprocess(PIL.Image.open(image_path)).unsqueeze(0).to(device)
        expected_prompt = f"This is a picture of a(n) {regex_e.search(image_path).group()}."

        with torch.no_grad():
            _image_features = model.encode_image(image).to(device)
            _text_features = model.encode_text(text).to(device)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            max_index = np.argmax(probs)
            prediction = prompts[max_index]
            prob = probs.flatten()[max_index]

            results["y"].append(indexed_prompts[expected_prompt])
            results["y_hat"].append(probs.squeeze())
            results["predicted_prompt"].append(prediction)
            results["probability"].append(prob)

    results_df = pd.DataFrame(results)
    results_df.to_pickle("inference_results.pk")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as e:
        logging.info("Inference completed.")
