import argparse
from configparser import ConfigParser
from src.utils import convert_config, data_check
from src.data_handler import DataHandler
from src.encoder import ImageEncoder
from src.decoder import CaptionDecoder
import tensorflow as tf
from src.training_manager import TrainingManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='data')
parser.add_argument('--config-path', type=str, dest='config_path', default='config.ini')
args = parser.parse_args()

config = ConfigParser()
config.read(args.config_path, encoding='utf-8')
config_base = convert_config(config['DEFAULT'])

data_handler = DataHandler(config_base, args.data_path)

if data_check(args.data_path):
    batched_dataset = data_handler.get_batched_ds()
else:
    batched_dataset = data_handler.get_batched_ds(download=True)

##### MODELS #######

encoder = ImageEncoder(config_base['embed_dim'])
decoder = CaptionDecoder(config_base['embed_dim'], config_base['attn_units'],
                         config_base['vocab_size'], data_handler.config['max_caption_len'])
optimizer = tf.keras.optimizers.Adam()
training_manager = TrainingManager(encoder, decoder, data_handler.tokenizer, optimizer, config_base)
training_manager.fit(batched_dataset)
