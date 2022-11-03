import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import json
from coordinates import *
import os

# ------------------------------------------------------------- Saving Results  -------------------------------------------------------------
class Encoder(json.JSONEncoder):
    """
    :param json.JSONEncoder: Extend the json.JSONEncoder to change data types for json files
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(transforms: np.ndarray, environment: Dict[str, Any], args,folder:str) -> None:
    """
    :param transforms: Transformation matrix
    :param environment: Environment dictionary taken from the yaml file
    :param args: Arguments taken from the command line when running the python file
    """
    # Serializing json
    json_object = json.dumps(transforms, indent=4, cls=Encoder)
    directory = f"{args.output_dir}/{environment['title']}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = f"{args.output_dir}/{environment['title']}/transforms_{folder}.json"
    with open(path, "w") as outfile:
        outfile.write(json_object)


def plot_images(
    img1: np.ndarray, img2: np.ndarray, title1: str = "", title2: str = ""
) -> None:
    """ Display a two column image 
    :param img1: First image
    :param img2: Second image
    :param title1: Title corresponding to first image
    :param title2: Title corresponding to first image
    :return: images and their titles
    """
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(121)  # 121 - 1 Row 2 Columns and Target 1st Column of Row
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)
    ax2 = fig.add_subplot(122)  # 121 - 1 Row 2 Columns and Target 1st Column of Row
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)


def plot_nimages(
    images: List[np.ndarray],
    image_titles: List[str],
    args,
    ignore: bool = False,
    cmap: bool = True,
    figsize: Tuple = (25, 15),
    path_title: str = "r",
    image_title: str = "random"
) -> None:
    """ Display an n column image 
    :param images: list of images
    :param image_titles: list of titles
    :param ignore: Display as grayscale image
    :param cmap: Use cmap
    :param figsize: Figure size
    """
    fig1, f1_axes = plt.subplots(
        ncols=len(images), nrows=1, constrained_layout=True, figsize=figsize
    )
    for i in range(len(images)):
        if not ignore:
            if not cmap:
                f1_axes[i].imshow(images[i])
                f1_axes[i].set_title("{}".format(image_titles[i]))
            else:
                f1_axes[i].imshow(images[i], cmap="gray")
                f1_axes[i].set_title("{}".format(image_titles[i]))
        else:
            f1_axes[i].imshow(images[i].astype(np.uint8), cmap="gray")
            f1_axes[i].set_title("{}".format(image_titles[i]))
    if not os.path.exists(f"{args.output_dir}/{path_title}"):
        os.makedirs(f"{args.output_dir}/{path_title}")
    fig1.savefig(f"{args.output_dir}/{path_title}/{image_title}.png")


def save_images(
    images: np.ndarray,
    environment: Dict[str, Any],
    args,
    image_titles: List[str] = None,
    path_title: str = "r",
    cmap: str = "viridis",
    image_type: str = None,
    image_title: str = "random"
) -> None:
    if image_titles is not None:
        plot_nimages(images, image_titles,args, path_title=path_title,image_title=image_title)
    else:
        if not os.path.exists(f"{args.output_dir}/{path_title}"):
            os.makedirs(f"{args.output_dir}/{path_title}")
        if path_title.split("/")[-1] == "test":
            if image_type is not None:
                for (index, image) in enumerate(images):
                    plt.imsave(f"{args.output_dir}/{path_title}/r_{index}_{image_type}.png", image,cmap=cmap)
            else:
                for (index, image) in enumerate(images):
                    plt.imsave(f"{args.output_dir}/{path_title}/r_{index}.png", image,cmap=cmap)
        else:   
            for (index, image) in enumerate(images):
                plt.imsave(f"{args.output_dir}/{path_title}/r_{index}.png", image,cmap=cmap)


# ------------------------------------------------------------- Manipulating Environment  -------------------------------------------------------------

def initial_rotation(location:np.ndarray,angle:float) -> np.ndarray:
    r"""
    :param location: Location of the camera in cartesian coordinates
    :param angle: Angle of rotation
    """
    location = cartesian_to_cylindrical(*location,True)
    location[1] = angle
    location = cylindrical_to_cartesian(*location,True)
    return location


def generate_locations(location:np.ndarray,end_angle:np.ndarray,num_samples: int = 40)->np.ndarray:
    start_angle = cartesian_to_cylindrical(*location,True)[1]
    if num_samples is None:
        angles = np.linspace(start_angle,end_angle,num=int(abs(start_angle)+abs(end_angle))+1)
    else:
        angles = np.linspace(start_angle,end_angle,num=num_samples)
    locations = []
    for angle in angles:
        location = cartesian_to_cylindrical(*location,True)
        location[1] = angle
        location = cylindrical_to_cartesian(*location,True)
        locations.append(location)
    return locations

def generate_angle(location:np.ndarray,angle:float,negative_direction:bool = True,angles:List[float]=None,locations:List[float]=None) -> Union[np.ndarray, Tuple[np.ndarray,List[float],List[float]]]:
    r"""
    :param location: Location of the camera in cartesian coordinates
    :param angle: Angle of rotation
    :param negative_direction: Whether we should rotate leftwards or rightwards
    :param angles: Preseve list of angles for debugging
    :param locations: Preseve list of camera locations in cartesian coordinates for debugging
    """
    location = cartesian_to_cylindrical(*location,True)
    location[1] = location[1] - angle if negative_direction else location[1] + angle
    if angles is not None:
        angles.append(location[1])
    location = cylindrical_to_cartesian(*location,True)
    if locations is not None:
        locations.append(location)
        return location,angles,locations
    else:
        return location