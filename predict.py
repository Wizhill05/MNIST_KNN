import os
import numpy as np
from PIL import Image
import joblib
import os
import shutil


def preprocess_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img).flatten()
    return img_array


def predict_using_images():
    model_filename = "knn_mnist_model.joblib"
    knn = joblib.load(model_filename)

    image_dir = r"images/"
    destination_dir = r"predictions/"

    try:
        image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        images = [
            preprocess_image(os.path.join(image_dir, file)) for file in image_files
        ]

        X_new = np.array(images)

        predictions = knn.predict(X_new)
        probabilities = knn.predict_proba(X_new)

        prediction_list = {}
        for file, pred, probs in zip(image_files, predictions, probabilities):
            temp = []
            print(f"Image: {file}, Predicted Label: {pred}")
            for digit, prob in enumerate(probs):
                temp.append(int(prob * 10000) / 100)
            print()
            prediction_list[file] = temp
            destination_path = os.path.join(
                destination_dir, os.path.basename(os.path.join(image_dir, file))
            )
            shutil.move(os.path.join(image_dir, file), destination_path)
        return prediction_list
    except ValueError:
        return []


def predict_from_grayscale_image(image_2d):
    model_filename = "knn_mnist_model.joblib"
    knn = joblib.load(model_filename)

    image_dir = r"images/"
    destination_dir = r"predictions/"

    try:
        image_array = np.array(image_2d).flatten().reshape(1, -1)
        prediction = knn.predict(image_array)[0]
        probabilities = knn.predict_proba(image_array)[0]

        probabilities_percent = [int(prob * 10000) / 100 for prob in probabilities]

        print(f"Predicted Label: {prediction}")
        print(f"Probabilities: {probabilities_percent}")

        return probabilities_percent
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return []
