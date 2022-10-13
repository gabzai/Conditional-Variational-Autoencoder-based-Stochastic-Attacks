from math import pi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, concatenate, Lambda

class Sampling(Layer):
    """Uses (v_mean, v_log_var) to sample z, the vector encoding a digit (reparametrization trick).
    This class is highly inspired from https://keras.io/examples/generative/vae/"""

    def call(self, inputs):
        """
        :param inputs: ([v_mean, v_log_var])
        :return: A set of samples z which follows the multivariate normal distribution N(mu, Sigma)
        """
        v_mean, v_log_var = inputs
        batch = tf.shape(v_mean)[0]
        dim = tf.shape(v_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return v_mean + tf.exp(0.5 * v_log_var) * epsilon

class VAE(Model):
    """This class is highly inspired from https://keras.io/examples/generative/vae/""" 
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Initialization of the learning metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Computation of the Elbo loss terms during the training process
        :param data: ([input traces, projected traces])
        :return: The result of each loss, namely Elbo/reconstruction/kl-divergence losses s.t. elbo = reconstruction + kl-divergence losses
        """
        with tf.GradientTape() as tape:
            # Computation of each term that are needed for the loss computation
            v_mean, v_log_var, v_sample = self.encoder([data[0][0], data[0][1]])
            reconstruction = self.decoder([v_sample, data[0][1]])

            # Computation of the reconstruction loss
            reconstruction_loss = tf.reduce_mean(0.5 * tf.math.log(2 * pi * 
                keras.losses.mse(data[0][0], reconstruction))/tf.math.log(2.0) + 0.5)

            # Computation of the KL-divergence loss
            kl_loss = -0.5 * (1 + v_log_var - tf.square(v_mean) - tf.exp(v_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Computation of the ELBO loss
            total_loss = reconstruction_loss + kl_loss

        # Computation and application of the gradient descent algorithm
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update of the loss values
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, val_data):
        """
        Computation of the validation loss terms during the training process
        :param val_data: ([input validation traces, projected validation traces])
        :return: The result of each validation loss, namely validation Elbo/reconstruction/kl-divergence losses s.t. elbo = reconstruction + kl-divergence losses
        """

        # Computation of each term that are needed for the validation loss computation
        val_v_mean, val_v_log_var, val_v_sample = self.encoder([val_data[0], val_data[1]])
        val_reconstruction = self.decoder([val_v_sample, val_data[1]])

        # Computation of the validation reconstruction loss
        val_reconstruction_loss = tf.reduce_mean(0.5 * tf.math.log(2 * pi * 
                keras.losses.mse(val_data[0], val_reconstruction))/tf.math.log(2.0) + 0.5)

        # Computation of the validation KL-divergence loss
        val_kl_loss = -0.5 * (1 + val_v_log_var - tf.square(val_v_mean) - tf.exp(val_v_log_var))
        val_kl_loss = tf.reduce_mean(tf.reduce_sum(val_kl_loss, axis=1))

        # Computation of the validation ELBO loss
        val_total_loss = val_reconstruction_loss + val_kl_loss

        # Update of the validation loss values
        self.total_loss_tracker.update_state(val_total_loss)
        self.reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.kl_loss_tracker.update_state(val_kl_loss)

        return val_total_loss

def define_encoder(input_size=1000, base_u=256):
    """
    Construction of the encoder
    :param input_size: Dimension of the input traces
    :param base_u: basis of the monomial subspace (F(1) => base_u=1 / F(2) => base_u=9 / F(3) => base_u=38 / F(4) => base_u=94 / F(5) => base_u=164 / F(6) => base_u=220 / F(7) => base_u=248 / F(8) => base_u=256 / F(9) => base_u=257)
    :return: Encoder model
    """

    # Input initialization (input traces, protected traces)
    input_shape1 = (input_size,)
    tr_input = Input(shape=input_shape1, name="trace")

    input_shape2 = (input_size, base_u)
    base_input = Input(shape=input_shape2, name="orthonormal_basis")

    # Extraction of the leakage model part (psi layer)
    psi_t = []
    for i in range(input_size):
        input_slice = Lambda(lambda x: x[:,i,:], name = "base_t" + str(i))(base_input)
        psi_t.append(Dense(1, activation=None, name="psiTheta_t"+str(i))(input_slice))
    psi_layer = concatenate(psi_t, name='psi_layer')

    # Noise extraction
    noise_layer = Lambda(lambda x: x[0] - x[1], name = "noise_estimation")([tr_input, psi_layer])

    # Output layers
    v_mean = Dense(input_size, activation=None, name = "v_mean")(noise_layer)
    v_log_sigma = Dense(input_size, activation=None, name = "v_log_sigma")(noise_layer)

    # Reparametrization trick
    v = Sampling()([v_mean, v_log_sigma])

    encoder = Model([tr_input, base_input], [v_mean, v_log_sigma, v], name="encoder")
    return encoder

def define_decoder(input_size=1000, base_u=256):
    """
    Construction of the decoder
    :param input_size: Dimension of the input latent representation
    :param base_u: basis of the monomial subspace (F(1) => base_u=1 / F(2) => base_u=9 / F(3) => base_u=38 / F(4) => base_u=94 / F(5) => base_u=164 / F(6) => base_u=220 / F(7) => base_u=248 / F(8) => base_u=256 / F(9) => base_u=257)
    :return: Decoder model
    """
    
    # Input initialization (input latent representation, protected traces)
    input_shape1 = (input_size,)
    latent_inputs = Input(shape=input_shape1, name="v") # v: Latent representation

    input_shape2 = (input_size, base_u)
    base_input = Input(shape=input_shape2, name="orthonormal_basis")

    # Construction of the synthetic traces (T synthetic = psi + v)
    psi_t = []
    for i in range(input_size):
        input_slice = Lambda(lambda x: x[:,i,:], name = "base_t" + str(i))(base_input)
        psi_t.append(Dense(1, activation=None, name="psiPhi_t"+str(i))(input_slice))
        
    psi_layer = concatenate(psi_t, name='psi_layer')
    synthetic_trace = Lambda(lambda x: x[0] + x[1], name = "synthetic_trace")([latent_inputs, psi_layer])

    decoder = Model([latent_inputs, base_input], synthetic_trace, name="decoder")
    return decoder