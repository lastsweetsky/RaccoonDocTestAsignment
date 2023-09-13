import json
import numpy as np
from PIL import Image


def preprocess_image(image):
    """
    Preprocesses the image by normalizing it and subtracting the mean.

    Args:
        image (np.ndarray): The image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image.
    """

    image = image / 255
    image -= 0.5
    return image.astype("float16")


def get_data(start=0, end=None, batch_size=None, channel=3, data_path="dataset/images"):
    """
    Get a batch of training data.

    Args:
        start (int, optional): The index of the first image in the batch. Defaults to 0.
        end (int, optional): The index of the last image in the batch. If `None`, then all images are used. Defaults to None.
        batch_size (int, optional): The size of the batch. Defaults to None.
        channel (int, optional): The number of channels in the images. Defaults to 3.
        data_path (str, optional): The path to the directory containing the images. Defaults to `dataset/images`.

    Returns:
        tuple: A tuple of the batch data and labels.
    """

    label = json.load(open("dataset/lables.json"))
    if not end:
        end = len(label)
    batch_size = end - start
    label = label[start:end]
    label = np.array(label).astype("float16").reshape((batch_size, 1)) / 30
    image_batch = np.zeros((batch_size, 200, 400, channel)).astype("float16")

    for i in range(start, end):
        image = Image.open("{}/{}.jpg".format(data_path, i))
        image = np.array(image)
        image_batch[i - start] = preprocess_image(image).reshape((200, 400, channel))

    return (image_batch, label)
