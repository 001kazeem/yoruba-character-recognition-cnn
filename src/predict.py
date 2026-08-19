import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Path to the trained model
MODEL_PATH = "models/mymodel.h5"


# Yoruba character class mapping
CLASS_DICT = {
    0: "A", 1: "B", 2: "D", 3: "E", 4: "F",
    5: "G", 6: "GB", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "R", 17: "S", 18: "T", 19: "U",
    20: "W", 21: "Y", 22: "a", 23: "b", 24: "d",
    25: "e", 26: "f", 27: "g", 28: "gb", 29: "h",
    30: "i", 31: "j", 32: "k", 33: "l", 34: "m",
    35: "n", 36: "o", 37: "p", 38: "r", 39: "s",
    40: "t", 41: "u", 42: "w", 43: "y", 44: "Ṣ",
    45: "ṣ", 46: "Ẹ", 47: "ẹ", 48: "Ọ", 49: "ọ"
}


def predict_character(image_path):
    """Predict the Yoruba character contained in an image."""

    # Load trained CNN
    model = load_model(MODEL_PATH)

    # Load and preprocess image
    img = image.load_img(
        image_path,
        target_size=(64, 64)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Find class with highest probability
    predicted_index = np.argmax(prediction[0])
    predicted_character = CLASS_DICT[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    print(f"Predicted character: {predicted_character}")
    print(f"Confidence: {confidence:.2f}%")

    return predicted_character, confidence


if __name__ == "__main__":
    image_path = input("Enter the path to the character image: ")
    predict_character(image_path)