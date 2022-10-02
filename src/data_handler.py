import tensorflow as tf
import tensorflow_datasets as tfds


class DataHandler:

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.tokenizer = None

    def get_batched_ds(self, data_type='val', download=False):
        dataset = self.get_data(data_type, download)
        captions = self.retrieve_captions(dataset)
        tokenizer = self.get_tokenizer(captions)
        batched_dataset = self.create_batched_ds(dataset, tokenizer)
        return batched_dataset

    def create_batched_ds(self, dataset, tokenizer):
        # generator that does tokenization, padding on the caption strings
        # and yields img, caption
        def generate_image_captions():
            for data in dataset:
                captions = data['captions']
                img_tensor = data['image_tensor']
                str_captions = [self.preprocess_caption(c) for c in captions.numpy()]
                padded = tokenizer(str_captions)
                for caption in padded:
                    yield img_tensor, caption  # repeat image

        return tf.data.Dataset.from_generator(
            generate_image_captions,
            (tf.float32, tf.int32)).batch(self.config['batch_size'])

    def get_data(self, data_type='val', download=False):
        dataset = tfds.load('coco_captions',
                            split=data_type,
                            shuffle_files=False,
                            download=download,
                            data_dir=self.data_path)
        dataset = dataset.map(self.get_image_label)
        return dataset

    def get_image_label(self, example):
        captions = example['captions']['text']  # all the captions
        img_id = example['image/id']
        img = example['image']
        img = tf.image.resize(img, (self.config['img_height'], self.config['img_width']))  # inception size
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return {
            'image_tensor': img,
            'image_id': img_id,
            'captions': captions
        }

    def preprocess_caption(self, c):
        global MAX_CAPTION_LEN
        MAX_CAPTION_LEN = 1
        caption = "starttoken {} endtoken".format(c.decode('utf-8'))
        words = [word for word in caption.lower().split()
                 if word not in self.config['stopwords']]
        MAX_CAPTION_LEN = max(MAX_CAPTION_LEN, len(words))
        self.config['max_caption_len'] = MAX_CAPTION_LEN
        return ' '.join(words)

    def retrieve_captions(self, dataset):
        captions = []
        for data in dataset:
            str_captions = [self.preprocess_caption(c) for c in data['captions'].numpy()]
            captions.extend(str_captions)
        return captions

    def get_tokenizer(self, captions):
        tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=self.config['vocab_size'], output_sequence_length=MAX_CAPTION_LEN)
        tokenizer.adapt(captions)
        self.tokenizer = tokenizer
        return tokenizer
