from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Activation,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras import optimizers


def build_model():
    """Build and compile the Yoruba character recognition CNN."""

    model = Sequential()

    # First convolutional block
    model.add(Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(64, 64, 3)
    ))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third convolutional block
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classification layers
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Output layer: 50 Yoruba character classes
    model.add(Dense(50, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model