import os
import re
import time

import tensorflow as tf

from src.decoder import CaptionDecoder


class TrainingManager:
    """
    Class that orchestrate the process of training model.
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

    def fit(self, batched_dataset) -> None:
        """
        A method that fits a passed batched dataset to the model. It iterates through data, performs a train_step method
        (defined above) and calculates loss. Each 100 batches it checks whether save_model (based on self.save_model_check
        method which checks its loss value). After each epoch, it's number, loss and time to complete is printed.
        Additionally, it uses early stopping (threshold is set in the config file)
        :param batched_dataset: Data to be fitted
        """
        prev_epoch_loss = 999
        loss_plot = []

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

            # storing the epoch end loss value to plot later
            loss_plot.append(current_loss)

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
            if loss_value < min(saved_models_losses):  # If current model is better than others
                return True
        return True  # No model was saved yet, so save the current one

