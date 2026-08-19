import os
import random
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Dataset configuration
DATADIR = os.path.join("dataset")

# Read the dataset CSV file
dataset = pd.read_csv(os.path.join(DATADIR, "yoruba character.csv"))

# Randomly select 500 images for validation
rand = random.sample(range(len(dataset)), 500)

validation_set = pd.DataFrame(
    dataset.iloc[rand, :].values,
    columns=["image", "label"]
)

# Remove validation samples from the training dataset
dataset.drop(rand, inplace=True)

# Randomly select 5 images from the validation set for testing
rand = random.sample(range(len(validation_set)), 5)

test_set = pd.DataFrame(
    validation_set.iloc[rand, :].values,
    columns=["image", "label"]
)

# Remove test samples from the validation set
validation_set.drop(rand, inplace=True)


# Image preprocessing and augmentation
train_data_generator = ImageDataGenerator(
    rescale=1 / 255,
    shear_range=0.2,
    zoom_range=0.2
)

data_generator = ImageDataGenerator(
    rescale=1 / 255
)


# Training data generator
training_data_frame = train_data_generator.flow_from_dataframe(
    dataframe=dataset,
    directory=DATADIR,
    x_col="image",
    y_col="label",
    target_size=(64, 64),
    class_mode="categorical"
)


# Validation data generator
validation_data_frame = data_generator.flow_from_dataframe(
    dataframe=validation_set,
    directory=DATADIR,
    x_col="image",
    y_col="label",
    target_size=(64, 64),
    class_mode="categorical"
)


# Test data generator
test_data_frame = data_generator.flow_from_dataframe(
    dataframe=test_set,
    directory=DATADIR,
    x_col="image",
    y_col="label",
    target_size=(64, 64),
    class_mode="categorical",
    shuffle=False
)