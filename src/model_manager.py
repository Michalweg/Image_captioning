import os
import re
import time

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from src.decoder import CaptionDecoder
from src.utils import prepare_image_for_model


class ModelManager:
    """
    Class that orchestrates the usage of a model.
    """

    def __init__(self, encoder, decoder, tokenizer, optimizer, config, saved_models_file_dir='saved_models'):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True, reduction='none')
        self.optimizer = optimizer
        self.config = config
        self.saved_models_file_dir = saved_models_file_dir

    def loss_function(self, real_values, pred):
        mask = tf.math.logical_not(tf.math.equal(real_values, 0))
        loss_ = self.loss_object(real_values, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    loss_plot = []

    @tf.function
    def train_step(self, img_tensor, target) -> tuple:
        """
        Function which performs one train step, from retrieving features from the encoder to generating texts with decoder.
        It also calculates gradients and updates model's parameters with them.
        :param img_tensor: Image from which texts are about to be generated
        :param target: Tokenized target captions
        :return: It outputs loss and total loss (loss divided by the len of the caption)
        """
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = CaptionDecoder.reset_state(target.shape[0], self.decoder.units)

        # initialize the batch of predictions with [[3],[3], ...] i.e. with start tokens
        dec_input = tf.expand_dims([self.tokenizer(['starttoken'])[0][0]] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder((dec_input, features, hidden))

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def fit(self, batched_dataset) -> dict:
        """
        A method that fits a passed batched dataset to the model. It iterates through data, performs a train_step method
        (defined above) and calculates loss. Each 100 batches it checks whether save_model (based on self.save_model_check
        method which checks its loss value). After each epoch, it's number, loss and time to complete is printed.
        Additionally, it uses early stopping (threshold is set in the config file)
        :param batched_dataset: Data to be fitted
        """
        prev_epoch_loss = 999
        history_dict = {"epochs": 0, "loss": []}

        for epoch in range(self.config['epochs']):
            start = time.time()
            total_loss = 0
            num_steps = 0

            for batch, (img_tensor, target) in enumerate(batched_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss
                num_steps += 1

                if batch % 100 == 0:
                    scaled_batch_los = batch_loss.numpy() / int(target.shape[1])
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, scaled_batch_los))

                    if self.save_model_check(scaled_batch_los):
                        self.save_model(scaled_batch_los)

            current_loss = total_loss / num_steps
            history_dict['epochs'] += 1
            history_dict['loss'].append(current_loss.numpy())

            print('Epoch {} Loss {:.6f} Time taken {:.1f} sec'.format(
                epoch + 1,
                current_loss,
                time.time() - start))

            # stop once it has converged
            improvement = prev_epoch_loss - current_loss
            if improvement < self.config['early_stop_thresh']:
                print("Stopping because improvement={} < {}".format(improvement, self.config['early_stop_thresh']))
                break
            prev_epoch_loss = current_loss

        return history_dict

    def save_model(self, loss_value) -> None:
        """
        Saves both encoder and decoder with regard to theirs corresponding losses.
        :param loss_value: Loss value corresponding to the models that are about to be saved
        """
        model_path = os.path.join(self.saved_models_file_dir, 'total_loss_' + str(round(loss_value, 3)))
        model_path = model_path if model_path[0] != '/' else model_path[1:]  # Permission problem
        self.encoder.save(os.path.join(model_path, 'encoder'))
        self.decoder.save(os.path.join(model_path, 'decoder'))

    def save_model_check(self, loss_value) -> bool:
        """
        Checks whether a model should be saved. It iterates through already saved models, checks whether the new loss
        value is lower than the found ones and if so returns a boolean value that tells to save a model
        :param loss_value: Loss value corresponding to the models that are about to be saved
        :return: A boolean value describing whether saving can be performed
        """
        try:
            saved_models_paths = os.listdir(self.saved_models_file_dir)
        except FileNotFoundError:  # The provided pass does not exist (saved_models dir has not been created yet)
            return True  # Creates saved_models directory and saves model

        saved_models_losses = []
        # Iterates through found model paths and checks whether a new model should be saved
        for saved_model_path in saved_models_paths:
            # List of numbers found in the saved model path
            saved_model_loss = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", saved_model_path)
            # Exclude any integers
            saved_model_loss = [model_loss for model_loss in saved_model_loss if '.' in model_loss]

            if saved_model_loss:
                saved_models_losses.append(float(saved_model_loss[0]))  # If loss is float, appends to the list

        if saved_models_losses:  # If any model already saved
            return loss_value < min(saved_models_losses)  # If current model is better than others

        return True  # No model was saved yet, so save the current one

    def predict(self, data_source) -> list[list]:
        """
        Method that predicts captions given data. Data_source can be either of a form of string (path to an image), or
        a tensor in the shape of [no. images, width, height, no, channels]
        :param data_source: Handled data sources: str (path to an image) tensor, or list of paths to which predictions
        should be made
        :return: List of lists (for each image one list)
        """
        if type(data_source) == str:
            generated_captions = self.predict_from_path(data_source)
        elif type(data_source) == EagerTensor:
            generated_captions = self.predict_from_tensor(data_source)
        elif type(data_source) == list:
            generated_captions = self.predict_form_paths_list(data_source)
        else:
            raise NotImplementedError(f"This data source: {data_source} is not yet implemented ")
        return generated_captions

    def predict_from_tensor(self, data) -> list[list]:
        """
        Iterates through images in a batch and creates predictions from images which are already preprocessed, in the
        form of tensor
        :param data: Preprocessed images in the form of a tensor with shape [no. images, width, height, no.channels]
        :return: List with captions for each image
        """
        all_images_outcome = []
        data = tf.expand_dims(data, 0) if len(data.shape) == 3 else data  # If just one image is passed
        for image in data:
            image = tf.expand_dims(image, 0) if len(image.shape) == 3 else image  # Add batch dim if needed
            image_captions = self.predict_image(image)
            all_images_outcome.append(image_captions)

        return all_images_outcome

    def predict_from_a_file(self, file_path) -> list[str]:
        """
        First it reads and preprocessses a file with prepare_image_for_model method and the generates captions with
        self.predict_image method
        :param file_path: File path to an image to generate captions for
        :return: List of generated captions for the passed image
        """
        image = prepare_image_for_model(file_path)
        captions = self.predict_image(image)
        return captions

    def predict_image(self, image) -> list[str]:
        """
        Creates captions for an image
        :param image: Image to generate captions for
        :return: List of generated captions
        """
        outcome = []
        features = self.encoder(image)
        hidden = CaptionDecoder.reset_state(1, self.decoder.units)
        dec_input = tf.expand_dims([self.tokenizer(['starttoken'])[0][0]], 0)

        for i in range(self.config['max_caption_len']):
            predictions, hidden, _ = self.decoder((dec_input, features, hidden))
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            outcome.append(self.tokenizer.get_vocabulary()[predicted_id])
            if self.tokenizer.get_vocabulary()[predicted_id] == 'endtoken':
                break
            dec_input = tf.expand_dims([predicted_id], 0)

        return outcome

    def predict_form_paths_list(self, paths_list:list) -> list[list]:
        generated_captions = []

        for file_path in paths_list:
            captions_for_image = self.predict_from_a_file(file_path)
            generated_captions.append(captions_for_image)

        return generated_captions

    def predict_from_path(self, path):
        generated_captions = []
        if os.path.isdir(path):
            for file_path in os.listdir(path):
                image_path = os.path.join(path, file_path)
                one_file_captions = self.predict_from_a_file(image_path)
                generated_captions.append(one_file_captions)
        else:
            generated_captions = self.predict_from_a_file(path)

        return generated_captions
