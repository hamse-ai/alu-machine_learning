#!/usr/bin/env python3
"""
Defines function that creates a variational autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Args:
        input_dims: integer containing dimensions of the model input
        hidden_layers: list containing number of nodes for each hidden layer
        latent_dims: integer containing dimensions of the latent space

    Returns:
        encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    # Build encoder hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent space
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z],
                          name='encoder')

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    # Build decoder hidden layers in reverse order
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(input_dims,
                                         activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    auto = keras.Model(encoder_inputs, outputs, name='vae')

    # Custom loss function for VAE
    def vae_loss(y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= input_dims

        # KL divergence loss
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - \
            keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
