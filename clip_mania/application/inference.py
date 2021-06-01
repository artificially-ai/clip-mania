import os

import numpy as np

import torch
import clip

import PIL

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string(name='test_dataset_path', default=None, help='Absolute path to the test dataset location.',
                    required=True)

flags.DEFINE_string(name='model_path', default=None, help='Absolute path where to load the model from.',
                    required=True)

# flags.register_validator('test_dataset_path',
#                          lambda value: type(value) is str and os.path.isdir(value),
#                          message='--test_dataset_path must be a valid directory.')
#
# flags.register_validator('model_path',
#                          lambda value: type(value) is str and os.path.isdir(value),
#                          message='--model_path must be a valid directory.')


def main(_args):
    classes = ["a bird", "a dog", "a cat", "a airplane"]
    test_dataset_path = FLAGS.test_dataset_path
    model_path = FLAGS.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    image = preprocess(PIL.Image.open(test_dataset_path)).unsqueeze(0).to(device)
    text = clip.tokenize(classes).to(device)

    with torch.no_grad():
        _image_features = model.encode_image(image).to(device)
        _text_features = model.encode_text(text).to(device)

        logits_per_image, logits_per_text = model(image, text)
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