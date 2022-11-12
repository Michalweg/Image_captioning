import argparse
from configparser import ConfigParser

import tensorflow as tf

from src.data_handler import DataHandler
from src.decoder import CaptionDecoder
from src.encoder import ImageEncoder
from src.model_manager import ModelManager
from src.utils import convert_config, data_check, load_models, check_model_path

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='data')
parser.add_argument('--config-path', type=str, dest='config_path', default='config.ini')
parser.add_argument('--continue_training', type=bool, dest='continue_training', default=False)
parser.add_argument('--model_path', type=str, dest='model_path', default='saved_models/total_loss_3.968')
args = parser.parse_args()
check_model_path(args)  # Checks whether passed arguments are valid

config = ConfigParser()
config.read(args.config_path, encoding='utf-8')
config_base = convert_config(config['DEFAULT'])

data_handler = DataHandler(config_base, args.data_path)

# Checks for downloaded data, if there's None, downloads it
if data_check(args.data_path):
    batched_dataset = data_handler.get_batched_ds()
else:
    batched_dataset = data_handler.get_batched_ds(download=True)

##### MODELS #######

if args.continue_training:  # If already built model should be used
    encoder, decoder = load_models(args.model_path)
else:  # Creates new encoder and decoder
    encoder = ImageEncoder(config_base['embed_dim'])
    decoder = CaptionDecoder(config_base['embed_dim'], config_base['attn_units'],
                             config_base['vocab_size'], data_handler.config['max_caption_len'])

optimizer = tf.keras.optimizers.Adam()
training_manager = ModelManager(encoder, decoder, data_handler.tokenizer, optimizer, config_base)
history = training_manager.fit(batched_dataset)
# data = [image for image, label in batched_dataset.take(1)][0]
data = 'data/my_own_images/'

captions_list = training_manager.predict(data)
print(captions_list)

