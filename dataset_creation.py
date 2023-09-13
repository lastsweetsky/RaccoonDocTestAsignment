import json
import cv2
import numpy as np


def open_json(json_path):
    """
    Opens the JSON file at the specified path and returns the data as a dictionary.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The data in the JSON file.
    """

    with open(json_path) as file:
        data = json.load(file)
    return data


def change_filepath_in_json(json_path="initial_data/responce.json"):
    """
    Changes the file paths in the JSON file to point to the images in the `dataset/images` directory.

    Args:
        json_path (str, optional): The path to the JSON file. Defaults to `initial_data/responce.json`.
    """
    initial_data = open_json(json_path)
    new_data = {}
    for key, value in initial_data.items():
        image_name = key.split("/")[1]
        new_key = "initial_data/images/" + image_name
        new_data[new_key] = value

    with open("dataset/formatted_initial_data.json", "w") as formatted_initial_data:
        json.dump(new_data, formatted_initial_data, indent=4)


def get_list_of_possible_angles(json_path):
    """
    Get a list of possible angles for each image in the JSON data.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary contains the following keys:
            * `Image_name`: The name of the image.
            * `Default_angle`: The default angle of the image.
            * `Possible_angles`: A list of possible angles for the image.
    """

    formatted_initial_data = open_json(json_path)
    possible_angles = []
    for key, value in formatted_initial_data.items():
        possible_angles.append(
            {
                "Image_name": key,
                "Default_angle": value,
                "Possible_angles": [angle - value for angle in list(range(-30, 31))],
            }
        )
    with open("dataset/angles_data.json", "w") as angles_data:
        json.dump(possible_angles, angles_data, indent=4)


def rotate_image(image, angle):
    """
    Rotates the image by the specified angle.

    Args:
        image (np.ndarray): The image to be rotated.
        angle (float): The angle to rotate the image by.

    Returns:
        np.ndarray: The rotated image.
    """

    image = np.array(image)
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        M,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated_image


def resize_image(image, shape, interpolation=cv2.INTER_AREA):
    """
    Resizes the image to the specified shape.

    Args:
        image (np.ndarray): The image to be resized.
        shape (tuple): The shape of the resized image.
        interpolation (int, optional): The interpolation method to be used. Defaults to `cv2.INTER_AREA`.

    Returns:
        np.ndarray: The resized image.
    """

    resized_image = cv2.resize(image, (shape, shape))
    return resized_image


def process_images(input_json):
    """
    Process the images in the JSON file and save them to the `dataset/images` directory.

    Args:
        input_json (str): The path to the JSON file.
    """

    image_controll = 0
    label = []
    initial_images = open_json(input_json)
    for initial_image in initial_images:
        image_path = initial_image["Image_name"]
        initial_angle = initial_image["Default_angle"]
        angles = initial_image["Possible_angles"]
        print("Created images for", image_path)
        try:
            image = cv2.imread(image_path)
            for angle in angles:
                rotated_image = rotate_image(image, angle=angle)
                resized_image = cv2.resize(rotated_image, (200, 400))
                coorect_angle = angle + initial_angle

                label.append(coorect_angle)

                cv2.imwrite(
                    "{}/{}.jpg".format("dataset/images/", image_controll), resized_image
                )
                image_controll = image_controll + 1
        except:
            pass

    label = [round(i, 2) for i in label]
    json.dump(label, open("dataset/lables.json", "w"))

