import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

from .vime_utils import mask_generator, pretext_generator


def vime_semi(x_train, y_train, x_unlab, x_test, parameters, p_m, K, beta, file_name):
    # Network parameters
    hidden_dim = parameters["hidden_dim"]
    batch_size = parameters["batch_size"]
    iterations = parameters["iterations"]

    # Basic parameters
    data_dim = x_train.shape[1]
    # label_dim = y_train.shape[1]

    # Divide training and validation sets (9:1)
    idx = np.random.permutation(len(x_train))
    train_idx = idx[: int(len(idx) * 0.9)]
    valid_idx = idx[int(len(idx) * 0.9) :]

    x_valid = x_train[valid_idx]
    y_valid = y_train[valid_idx]

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    # Predictor model
    def create_predictor_model():
        model = models.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu", input_shape=(data_dim,)),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    predictor_model = create_predictor_model()

    # Compile the model
    predictor_model.compile(
        optimizer=optimizers.Adam(),
        # loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    # Load encoder from self-supervised model
    encoder = tf.keras.models.load_model(file_name)

    # Encode validation and testing features
    x_valid_encoded = encoder.predict(x_valid)
    x_test_encoded = encoder.predict(x_test)

    # Training loop
    best_loss = float("inf")
    no_improvement_counter = 0

    for it in range(iterations):
        # Select a batch of labeled data
        batch_idx = np.random.choice(len(x_train), batch_size, replace=False)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Encode labeled data
        x_batch_encoded = encoder.predict(x_batch)

        # Train the model
        predictor_model.train_on_batch(x_batch_encoded, y_batch)

        # Early stopping logic
        val_loss = predictor_model.evaluate(x_valid_encoded, y_valid, verbose=0)
        val_loss, val_accuracy = predictor_model.evaluate(
            x_valid_encoded, y_valid, verbose=0
        )
        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement_counter = 0
            # Save the best model
            predictor_model.save("best_predictor_model.h5")
        else:
            no_improvement_counter += 1
            if no_improvement_counter > 100:
                break

    # Load the best model
    best_model = tf.keras.models.load_model("best_predictor_model.h5")

    # Predict on x_test
    y_test_hat = best_model.predict(x_test_encoded)

    return y_test_hat
