import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image


class Racoon:
    """
    This class is used to detect the angle in an image and rotate the image accordingly.

    Attributes:
        model: The TensorFlow model used to detect the angle of the image.
    """

    def __init__(self):
        """
        Initializes the Racoon class.
        """
        self.load_model()

    def load_model(self):
        """
        Loads pretrained model.
        """
        self.model = tf.saved_model.load("saved_model")

    def preprocess_image(self, image):
        """
        Preprocees the image.

         Args:
            image: The numpy array of image.
        """
        image = image / 255
        image -= 0.5
        return image.astype("float32")

    def detect_angle(self, image) -> float:
        """
        Detects the angle in the specified image.

        Args:
            image: The image to detect the angle of.

        Returns:
            The angle in degrees.
        """
        image = image.resize((200, 400))
        image = np.array(image)
        preprocessed_image = self.preprocess_image(image).reshape((1, 200, 400, 3))

        # Use the loaded model for inference
        inference_fn = self.model.signatures["serving_default"]

        # Predict the angle using the loaded model
        angle_prediction = inference_fn(
            tf.constant(preprocessed_image, dtype=tf.float32)
        )
        # Access the output tensor by its name
        output_tensor_name = "dense_14"

        # Use tf.cast to convert to float32
        angle_prediction = tf.cast(
            angle_prediction[output_tensor_name], dtype=tf.float32
        )

        # Calculate the angle in degrees based on the model's output
        angle = round(angle_prediction[0][0].numpy() * 30, 2)
        print("Predicted angle:", angle)
        return angle

    def rotate_image(self, angle, image):
        """
        Rotates the specified image by defined angle over centre.

        Args:
            angle: The angle in degrees.
            image: The image to rotate.

        Returns:
            Rotated image.
        """
        image = np.array(image)
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(
            image,
            M,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated_image

    def image_name(self, image):
        """
        Defines image's filename and extension.

        Args:
            image: The image that was rotated.

        Returns:
            Filename and extension of image.
        """
        filename, extension = os.path.splitext(image.filename)
        return filename, extension

    def save_result(self, image, filename, extension):
        """Saves the rotated image to a file.

        Args:
            image: The rotated image.

        """
        filename = f"{filename}_result{extension}"
        path = os.path.join(os.getcwd(), filename)
        cv2.imwrite(path, image)

    def tranform(self, image, to_save: bool = False):
        """
        Rotates the specified image by the angle predicted by the model.

        Args:
            image: The image to rotate.
            to_save: Whether to save the rotated image to a file.

        Returns:
            The rotated image.
        """
        angle = self.detect_angle(image=image)
        rotated_image = self.rotate_image(angle=angle, image=image)
        if to_save:
            filename, extension = self.image_name(image=image)
            self.save_result(
                image=rotated_image, filename=filename, extension=extension
            )

        return rotated_image


def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB mode.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image: The loaded image.
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from '{image_path}': {str(e)}")
        return None

#

