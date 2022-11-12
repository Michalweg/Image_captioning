import ast
import os

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from src.decoder import CaptionDecoder
from src.encoder import ImageEncoder


def convert_config(parser) -> dict:
    """
    Converts parser, so it includes data in the right format
    :param parser: Args parser form which data is about to be converted
    :return: Converted argument parser
    """
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


def data_check(data_path: str) -> bool:
    """
    Check whether provided path includes some data
    :param data_path: Path to data to be checked
    :return: A boolean value whether provided path contains data
    """
    try:
        return os.listdir(data_path) is not None
    except FileNotFoundError:
        return False


def load_models(models_dir: str) -> (ImageEncoder, CaptionDecoder):
    """
    Loads method from provided directory
    :param models_dir: Path to the dir from which models are loaded
    :return: Tuple with loaded encoder and decoder
    """
    encoder = tf.keras.models.load_model(os.path.join(models_dir, 'encoder'), compile=False,
                                         custom_objects={'ImageEncoder': ImageEncoder})
    decoder = tf.keras.models.load_model(os.path.join(models_dir, 'decoder'), compile=False,
                                         custom_objects={'CaptionDecoder': CaptionDecoder})
    return encoder, decoder


def check_model_path(args):
    """
    Checks the correctness of provided path to a model.
    :param args: Passed arguments as arguments parser
    :return: Updated arguments (if needed)
    """
    if args.continue_training:
        if os.path.isdir(args.model_path):
            if os.listdir(args.model_path):
                return args
            raise FileNotFoundError(f"The provided path: {args.model_path} does not contain any files")
        raise FileNotFoundError(f"The provided path: {args.model_path} is invalid")


def prepare_image_for_model(file_path):
    """
    Reading and resizing an image so encoder's backbone model (inception_v3) can digest it
    :param file_path: Path to a file
    :return: Preprocessed image ready to be digested by a model
    """
    image_tensor = read_image_from_path(file_path)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = tf.keras.applications.inception_v3.preprocess_input(image_tensor)
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor


def read_image_from_path(file_path: str) -> EagerTensor:
    img_converter = {'.jpeg': tf.io.decode_jpeg,
                     '.png': tf.io.decode_png,
                     '.bmp': tf.io.decode_bmp}

    image_extension = os.path.splitext(file_path)[1]
    if image_extension not in img_converter.keys():
        raise NotImplementedError(f"{image_extension} is not yet supported")

    image_file = tf.io.read_file(file_path)
    image_tensor = img_converter[image_extension](image_file)

    return image_tensor
