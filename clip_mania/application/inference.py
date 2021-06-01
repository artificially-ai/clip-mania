import os

import numpy as np

import torch
import clip

from absl import app, flags, logging
from absl.flags import FLAGS
from torch.utils.data import DataLoader

from clip_mania.utils.data.datasets import FewShotDataset
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
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    indexed_prompts = DatasetProcessor.create_indexed_prompts(test_dataset_path)
    classes = list(indexed_prompts.values())

    dataset = FewShotDataset(test_dataset_path)
    test_dataloader = DataLoader(dataset, batch_size=len(classes), drop_last=True)
    for batch in test_dataloader:
        images, labels = batch
        for image, label in zip(images, labels):
            with torch.no_grad():
                _image_features = model.encode_image(image).to(device)
                _text_features = model.encode_text(label).to(device)

                logits_per_image, logits_per_text = model(image, label)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                max_index = np.argmax(probs)
                prediction = classes[max_index]
                prob = probs.flatten()[max_index]

                logging.info(f"Max probability is: {prob}")
                logging.info(f"Predicted class is: {prediction}")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as e:
        logging.info("Inference completed.")
