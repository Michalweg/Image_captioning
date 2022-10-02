import ast
import os

import tensorflow as tf

from src.decoder import CaptionDecoder
from src.encoder import ImageEncoder


def convert_config(parser):
    output = {}
    for key in parser:
        try:
            output[key] = int(parser[key])  # try if it is int
        except ValueError:
            try:
                output[key] = float(parser[key])  # try if it is float
            except ValueError:
                try:  # check if it is list
                    output[key] = ast.literal_eval(parser[key])
                except ValueError:
                    output[key] = parser[key]  # it is string
    return output


def data_check(data_path):
    try:
        if os.listdir(data_path) is not None:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def load_models(models_dir):
    encoder = tf.keras.models.load_model(os.path.join(models_dir, 'encoder'), compile=False,
                                         custom_objects={'ImageEncoder': ImageEncoder})
    decoder = tf.keras.models.load_model(os.path.join(models_dir, 'decoder'), compile=False,
                                         custom_objects={'CaptionDecoder': CaptionDecoder})
    return encoder, decoder


def check_args(args):
    """
    Checks the correctness of passed arguments
    :param args: passed arguments
    :return: Updated arguments (if needed)
    """
    if args.continue_training:
        args.continue_training = os.path.isdir(args.model_path)
    return args
