import os

import torch

from absl import app, flags, logging
from absl.flags import FLAGS

from clip_mania.core.executor import ModelExecutor

flags.DEFINE_string(name='dataset_path', default=None, help='Absolute path to the dataset location.', required=True)

flags.DEFINE_integer(name='epochs', default=None, help='Number of epochs to be trained on.', required=True)

flags.register_validator('dataset_path',
                         lambda value: type(value) is str and os.path.isdir(value),
                         message='--dataset_path must be a valid directory.')

flags.register_validator('epochs',
                         lambda value: type(value) is int,
                         message='--epochs must be an int value')


def main(_args):
    dataset_path = FLAGS.dataset_path
    epochs = FLAGS.epochs

    executor = ModelExecutor()
    model, preprocess = executor.train(dataset_path, epochs=epochs)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, "model_checkpoint/saved_model.pt")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as e:
        logging.error(f"Error occurred: {e}")
