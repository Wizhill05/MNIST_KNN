import os
import numpy as np
from PIL import Image
import joblib
import shutil


def preprocess_image(image_path):
    """
    Preprocess an image by converting it to grayscale, resizing it to 28x28, and flattening it into a 1D array.
    """
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img).flatten()
    return img_array


def predict_using_images():
    """
    Predict the labels of images in a directory using a pre-trained KNN model.
    """
    model_filename = "knn_mnist_model.joblib"
    knn = joblib.load(model_filename)

    image_dir = r"images/"
    destination_dir = r"predictions/"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(destination_dir, exist_ok=True)

    try:
        # Get a list of image files in the directory
        image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Preprocess each image and store them in a list
        images = [
            preprocess_image(os.path.join(image_dir, file)) for file in image_files
        ]

        # Convert the list of images to a NumPy array
        X_new = np.array(images)

        # Make predictions using the KNN model
        predictions = knn.predict(X_new)
        probabilities = knn.predict_proba(X_new)

        # Create a dictionary to store the predictions and probabilities for each image
        prediction_list = {}
        for file, pred, probs in zip(image_files, predictions, probabilities):
            temp = []
            print(f"Image: {file}, Predicted Label: {pred}")
            for digit, prob in enumerate(probs):
                temp.append(int(prob * 10000) / 100)
            print()
            prediction_list[file] = temp

            # Move the image to the destination directory
            destination_path = os.path.join(
                destination_dir, os.path.basename(os.path.join(image_dir, file))
            )
            shutil.move(os.path.join(image_dir, file), destination_path)

        return prediction_list
    except ValueError:
        return []


def predict_from_grayscale_image(image_2d):
    """
    Predict the label of a grayscale image using a pre-trained KNN model.
    """
    model_filename = "knn_mnist_model.joblib"
    knn = joblib.load(model_filename)

    try:
        # Flatten the 2D image array into a 1D array
        image_array = np.array(image_2d).flatten().reshape(1, -1)

        # Make a prediction using the KNN model
        prediction = knn.predict(image_array)[0]
        probabilities = knn.predict_proba(image_array)[0]

        # Convert the probabilities to percentages
        probabilities_percent = [int(prob * 10000) / 100 for prob in probabilities]

        print(f"Predicted Label: {prediction}")
        print(f"Probabilities: {probabilities_percent}")

        return probabilities_percent
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return []
