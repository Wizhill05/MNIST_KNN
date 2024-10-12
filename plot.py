import numpy as np
import matplotlib.pyplot as plt


def display_image(arr):
    # Reshape the array into a 28x28 image
    img = np.reshape(arr, (28, 28))

    # Normalize the pixel values to be between 0 and 1
    img = img / 255.0

    # Display the image using matplotlib
    plt.imshow(img, cmap="gray")
    plt.show()


# Example usage:
arr = []
display_image(arr)
