import numpy as np
import struct
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def load_external_data(csv_filename):
    df = pd.read_csv(csv_filename)
    # Assuming the CSV has columns 'label' and 'pixel0', 'pixel1', ..., 'pixel783' for 28x28 images
    labels = df["label"].values
    images = df.drop("label", axis=1).values.reshape(-1, 28, 28)
    return images, labels


def train():
    train_images_path = r"archive\train-images.idx3-ubyte"
    train_labels_path = r"archive\train-labels.idx1-ubyte"
    test_images_path = r"archive\t10k-images.idx3-ubyte"
    test_labels_path = r"archive\t10k-labels.idx1-ubyte"
    external_data_path = r"externals.csv"

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    external_images, external_labels = load_external_data(external_data_path)

    combined_images = np.concatenate((train_images, external_images), axis=0)
    combined_labels = np.concatenate((train_labels, external_labels), axis=0)

    X_train = combined_images.reshape(combined_images.shape[0], -1)
    y_train = combined_labels
    X_test = test_images.reshape(test_images.shape[0], -1)
    y_test = test_labels

    knn = KNeighborsClassifier(n_neighbors=245)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    model_filename = "knn_mnist_model.joblib"
    joblib.dump(knn, model_filename)

    accuracy = accuracy_score(y_test, y_pred)
    return (
        f"Accuracy: {accuracy * 100:.2f}%"
        + "\n"
        + "\nClassification Report:\n"
        + "\n"
        + classification_report(y_test, y_pred)
    )


print(train())
