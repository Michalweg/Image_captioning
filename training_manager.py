import tensorflow as tf
import time


class TrainingManager:

    def __init__(self, encoder, decoder, tokenizer, optimizer, config):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True, reduction='none')
        self.optimizer = optimizer
        self.config = config

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    loss_plot = []

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        # initialize the batch of predictions with [[3],[3], ...] i.e. with start tokens
        dec_input = tf.expand_dims([self.tokenizer(['starttoken'])[0][0]] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def fit(self, batched_dataset):
        prev_loss = 999
        loss_plot = []
        for epoch in range(20):
            start = time.time()
            total_loss = 0
            num_steps = 0

            for batch, (img_tensor, target) in enumerate(batched_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss
                num_steps += 1

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

            current_loss = total_loss / num_steps

            # storing the epoch end loss value to plot later
            loss_plot.append(current_loss)

            # ckpt_manager.save()

            print('Epoch {} Loss {:.6f} Time taken {:.1f} sec'.format(
                epoch + 1,
                current_loss,
                time.time() - start))

            # stop once it has converged
            improvement = prev_loss - current_loss
            if improvement < self.config['early_stop_thresh']:
                print("Stopping because improvement={} < {}".format(improvement, self.config['early_stop_thresh']))
                break
            prev_loss = current_loss

