import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import BatchDataset, MapDataset
from tensorflow.python.keras.layers.preprocessing.text_vectorization import TextVectorization


class DataHandler:
    """
    Class to retrieve and prepare data for the model ti digest.
    """

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.tokenizer = None

    def get_batched_ds(self, data_type='val', download=False) -> BatchDataset:
        dataset = self.get_data(data_type, download)
        captions = self.retrieve_captions(dataset)
        tokenizer = self.get_tokenizer(captions)
        batched_dataset = self.create_batched_ds(dataset, tokenizer)
        return batched_dataset

    def create_batched_ds(self, dataset, tokenizer) -> BatchDataset:
        """
        Batch the passed dataset from the generator defined below
        :param dataset: Dataset to be batched
        :param tokenizer: Adapted tokenizer
        :return: Batched dataset created from the generator
        """
        # Generator that does tokenization, padding on the caption strings
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

    def get_data(self, data_type='val', download=False) -> MapDataset:
        """
        Retrieves the 'coco_captions' dataset from tensorflow datasets. If the data has been already downloaded, it uses
        is, otherwise it downloads data. Once data is obtained, is being mapped to the desired format.
        :param data_type: Either train or val. Due to the fact, that the first step was to run the model instead of its
        perfromance, the default value is 'val' due to computation time
        :param download: Parameter that tells whether downloads data or not
        :return: Mapped dataset (in the form of {img_tensor, img_id, captions}
        """
        dataset = tfds.load('coco_captions',
                            split=data_type,
                            shuffle_files=False,
                            download=download,
                            data_dir=self.data_path)
        dataset = dataset.map(self.get_image_label)
        return dataset

    def get_image_label(self, example) -> dict:
        """
        Retrieves from an example from the dataset the image, image_id and captions. Additionally, image is being resized
        and preprocessed so it can be digested by the backbone model (inception in this example)
        :param example: An example from the dataset
        :return: Dictionary in that contains image, img_id and captions related to this image
        """
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

    def preprocess_caption(self, caption) -> str:
        """
        :param caption: Caption related to the image
        :return: Preprocessed caption (with start and end token, lowercase and without stop words)
        """
        max_caption_len = 1
        caption = "starttoken {} endtoken".format(caption.decode('utf-8'))
        words = [word for word in caption.lower().split()
                 if word not in self.config['stopwords']]
        max_caption_len = max(max_caption_len, len(words))
        self.config['max_caption_len'] = max_caption_len
        return ' '.join(words)

    def retrieve_captions(self, dataset) -> list[str]:
        """
        Method retrieves and preprocesses captions from the dataset
        :param dataset: Dataset which included captions
        :return: A list with preprocessed captions
        """
        captions = []
        for data in dataset:
            str_captions = [self.preprocess_caption(c) for c in data['captions'].numpy()]
            captions.extend(str_captions)
        return captions

    def get_tokenizer(self, captions) -> TextVectorization:
        """
        Obtains pretrained tokenizer from the tensorflow and adapts it to the captions
        :param captions: Captions that tokenizer adapts to
        :return: Adapted tokenizer
        """
        tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=self.config['vocab_size'], output_sequence_length=self.config['max_caption_len'])
        tokenizer.adapt(captions)
        self.tokenizer = tokenizer
        return tokenizer
