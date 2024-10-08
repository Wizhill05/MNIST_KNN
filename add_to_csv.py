import os
import json
import numpy as np
import pandas as pd
from PIL import Image


def process_images(image_dir, json_path, csv_path):
    # Load the JSON file
    with open(json_path, "r") as f:
        image_labels = json.load(f)

    # Initialize a list to store image data and labels
    data = []

    # Process each image in the directory
    for image_name in os.listdir(image_dir):
        # Check if the image is in the JSON mapping
        if image_name in image_labels:
            # Load the image
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            image_data = np.array(
                image
            ).flatten()  # Flatten the 28x28 image to a 784-length vector

            # Get the label from the JSON mapping
            label = image_labels[image_name]

            # Append the label and image data to the list
            data.append([label] + image_data.tolist())

            # Remove the entry from the JSON mapping
            del image_labels[image_name]

    # Save the updated JSON file
    with open(json_path, "w") as f:
        json.dump(image_labels, f, indent=4)

    # Create a DataFrame
    if data:
        columns = ["label"] + [f"pixel{i}" for i in range(784)]
        df = pd.DataFrame(data, columns=columns)

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            # Append to the existing CSV file without writing the header
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            # Write the CSV file with the header
            df.to_csv(csv_path, mode="w", header=True, index=False)

        print(f"Data successfully written to {csv_path}")
    else:
        print("No data to write to CSV.")


image_directory = "predictions/"
json_file_path = "new_data.json"
output_csv_path = "externals.csv"
process_images(image_directory, json_file_path, output_csv_path)
