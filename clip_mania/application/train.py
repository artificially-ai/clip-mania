import os

from absl import app, flags, logging
from absl.flags import FLAGS

from clip_mania.core.executor import ModelExecutor

flags.DEFINE_string(name='dataset_path', default=None, help='Absolute path to the dataset location.',
                    required=True)

flags.DEFINE_string(name='model_output_path', default=None,  help='Absolute path where to save the model.',
                    required=True)

flags.DEFINE_integer(name='batch_size', default=8,  help='The batch size. Use the number of classes you have.')

flags.DEFINE_integer(name='epochs', default=2, help='Number of epochs to be trained on.')

flags.DEFINE_string(name='model_name', default="clip_mania_model.pt",  help='Model file name.')

flags.register_validator('dataset_path',
                         lambda value: type(value) is str and os.path.isdir(value),
                         message='--dataset_path must be a valid directory.')

flags.register_validator('model_output_path',
                         lambda value: type(value) is str and os.path.isdir(value),
                         message='--model_output_path must be a valid directory.')

flags.register_validator('batch_size',
                         lambda value: type(value) is int,
                         message='--batch_size must be an int value')

flags.register_validator('epochs',
                         lambda value: type(value) is int,
                         message='--epochs must be an int value')


def main(_args):
    dataset_path = FLAGS.dataset_path
    model_output_path = FLAGS.model_output_path
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    model_name = FLAGS.model_name

    executor = ModelExecutor(batch_size=batch_size, lr=1e-8, weight_decay=0.1)
    model, preprocess = executor.train(dataset_path, epochs=epochs)
    ModelExecutor.save_model(model, model_output_path, model_name)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as e:
        logging.info("Training completed.")
