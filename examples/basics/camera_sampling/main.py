import blenderproc as bproc
from blenderproc.python.types.LightUtility import Light
import numpy as np
import argparse
import debugpy
import sys
import yaml
path_files = "/home/molefe/Playground/Research/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Constrain Dataset/version1/BlenderProc/examples/basics/camera_sampling"
sys.path.append(path_files)
from coordinates import *
from utils import *



# Run script: blenderproc run examples/basics/camera_sampling/main.py examples/resources/scene.obj examples/basics/camera_sampling/output --blender-install-path ./
# ------------------------------------------------------------- Debug  -------------------------------------------------------------

debugpy.listen(5678)
debugpy.wait_for_client()


# ------------------------------------------------------------- Arguments  -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "scene",
    nargs="?",
    default="examples/resources/Marina-converted.obj",
    help="Path to the scene.obj file",
)
parser.add_argument(
    "output_dir",
    nargs="?",
    default="examples/basics/camera_sampling/output",
    help="Path to where the final files, will be saved",
)
args = parser.parse_args()


# ------------------------------------------------------------- Envrionment  -------------------------------------------------------------
stream = open("environment.yaml","r")
environment = yaml.load(stream,Loader=yaml.FullLoader)
lighting = environment['lighting']
camera = environment['camera']
obj = environment['obj']



# ------------------------------------------------------------- Find Positions  -------------------------------------------------------------

# Testing: rgb image data type and shape
def test_default_position():
    """ Takes a set of positions for the camera and saves a row of images with titles describing their values
    Goal: Find vectors that correspond to frontal facing angles 
    :param num_samples: The number of images that you want to generate
    :param interval: How much farther the rotation should stretch, i.e 2.15 + i(2) will rotate two times as much
    :param random_sample: 
    :param location: If location is none then sample from random distributions. This helps us decide where our x,y,z should generally be for our dataThe number of images that you want to generate
    We will use location if we are certain of our x,y,z coordinates and want to sample
    :param chosen_index: The coordinate we would like to change with each came
    ra sample. chosen_index = 0 means we are varying the x coordinate
    :return: images and their titles
    """
    bproc.init()

    # load the objects into the scene
    blender_object = args.scene.split("/")[-1].split(".")[-1]
    if blender_object == 'obj':
        objs = bproc.loader.load_obj(args.scene)
    elif blender_object == 'blend':
        objs = bproc.loader.load_blend(args.scene)
    objs[0].set_scale(obj['scale'])

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type(lighting['light_type'])
    light.set_location(lighting['light_location'])
    light.set_energy(lighting['light_energy'])

    # Find point of interest, all cam poses should look towards it
    poi = bproc.object.compute_poi(objs)
    location_coordinates = []
    light_coordinates = []
    camera_random_uniform = camera['random_uniform']
    for _ in range(int(environment['num_samples'])):
        if environment['path'] == 'light':
            light = bproc.types.Light()
            light.set_type(lighting['light_type'])
            light_coordinate = np.random.uniform(lighting['locations']['low'],lighting['locations']['high'])
            light_coordinates.append(light_coordinate)
            light.set_location(light_coordinate)
            light.set_energy(lighting['light_energy'])
        if environment['path'] == 'light':
            random_uniform = camera['location']
        else:
            random_uniform = np.random.uniform(camera_random_uniform['low'],camera_random_uniform['high'])
        location_coordinates.append(random_uniform)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - random_uniform
        )  # I want pitch matrix
        cam2world_matrix = bproc.math.build_transformation_mat(
            random_uniform, rotation_matrix
        )  # (3x3)
        bproc.camera.add_camera_pose(cam2world_matrix)
        bproc.camera.set_intrinsics_from_blender_params(lens=float(obj['focal_length']), lens_unit="FOV") 

    normals = obj['normals']
    depth_map = obj['depth_map']
    if normals:
        bproc.renderer.enable_normals_output()
    if depth_map['depth']:
        bproc.renderer.enable_depth_output(activate_antialiasing=depth_map['antialiasing'])

    data = bproc.renderer.render()
    images = [data["colors"][i] for i in range(len(data["colors"]))]
    if environment['path'] == 'light':
        image_titles = [
            f"x={round(light_coordinate[0],2)}y={round(light_coordinate[1],2)},z={round(light_coordinate[2],2)}"
            for light_coordinate in light_coordinates
        ]
    else:
        image_titles = [
            f"x={round(location_coordinate[0],2)}y={round(location_coordinate[1],2)},z={round(location_coordinate[2],2)}"
            for location_coordinate in location_coordinates
        ]
    titles__ = dict(zip([i for i in range(len(image_titles))],image_titles))
    folder = environment['path']
    path_title = environment['title']
    save_path = f"{path_title}/{folder}"
    if environment['h5']:
        bproc.writer.write_hdf5(args.output_dir, data)
    save_images(images,environment,args,titles__,path_title=save_path)



# ------------------------------------------------------------- Generating Dataset  -------------------------------------------------------------
def generate_dataset():
    """ Generate NeRF dataset
    Goal: Find vectors that correspond to frontal facing angles 
    :param args: Terminal arguments
    :param focal_length: intrinsic parameter controlling focal view
    :param num_samples: The number of images that you want to generate
    :param random_sample: 
    :param location: If location is none then sample from random distributions. This helps us decide where our x,y,z should generally be for our dataThe number of images that you want to generate
    We will use location if we are certain of our x,y,z coordinates and want to sample
    :param chosen_index: The coordinate we would like to change with each camera sample. chosen_index = 0 means we are varying the x coordinate
    :return: images and their titles
    """
    
    path_title = environment['title']

    # load the objects into the scene
    
    initial_location = np.array(camera['location'])
    folders = ["train","valid","test"]
    transforms = {'intial_camera_location':initial_location}
    if camera['initial']['rotate']:
        initial_location = initial_rotation(initial_location,camera['initial']['angle'])
    end_angle = camera['rotation']['degrees']
    num_samples = None if environment['num_samples'] == 0 else environment['num_samples']
    locations = generate_locations(initial_location,end_angle,num_samples)  
    train,valid,test = environment['split']
    if environment['path'] == "dataset":
        split_locations = get_split_locations(locations,[train,valid,test])
        for folder,split_location in zip(folders,split_locations):
            generate_images(split_location,transforms,path_title,folder)
    else:
        folder = environment['path']
        generate_images(locations,transforms,path_title,folder)
    print("Done")
    
def get_split_locations(locations:np.ndarray,splits:List[int])->Tuple:
    r"""
    :param locations: Camera locations, in cartesian coordinates, that are samples from varying rotations around the z-axis
    :param splits: Dataset splits, train, valid, test
    """
    train,valid,test = [int((split/100) * len(locations)) for split in splits]
    np.random.shuffle(locations)
    train_locations = locations[:train]
    valid_locations = locations[train:train+valid]
    test_locations = locations[train+valid:train+valid+test]
    return train_locations, valid_locations, test_locations

def generate_images(locations:List[float],transforms:Dict[str,Any],path_title,folder:str = "train")->None:
    r"""
    :param poi: Point of interest of the object defined in cartesian coordinates,
    :param locations: camera locations defined in a list
    :param transforms: Dictionary that will be saved onto a JSON file
    :param path_title: Usually name of the model, i.e marina
    :param folder: Describes what kind of folder we are saving in, i.e folder="train"
    """
    bproc.init()
    blender_object = args.scene.split("/")[-1].split(".")[-1]
    if blender_object == 'obj':
        objs = bproc.loader.load_obj(args.scene)
    elif blender_object == 'blend':
        objs = bproc.loader.load_blend(args.scene)
    objs[0].set_scale(obj['scale'])

    # Find point of interest, all cam poses should look towards it
    poi = bproc.object.compute_poi(objs)
    frames = []
    for index,location in enumerate(locations):
        frame = {}
        light = bproc.types.Light()
        light.set_type(lighting['light_type'])
        light.set_location([location[0],location[1],location[2]+2])
        light.set_energy(lighting['light_energy'])
        
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location )  
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)  
        bproc.camera.add_camera_pose(cam2world_matrix)
        file_path = f"./{folder}/r_{index}"
        frame["file_path"] = file_path
        frame['transform_matrix'] = cam2world_matrix
        frames.append(frame)
        bproc.camera.set_intrinsics_from_blender_params(lens=float(obj['focal_length']), lens_unit="FOV")        
    save_path = f"{path_title}/{folder}"
    transforms['camera_angle_x'] = float(obj['focal_length'])
    transforms['frames'] = frames
    save_json(transforms,environment,args,folder)
    if folder == "test":
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()
    images = [data["colors"][i] for i in range(len(data["colors"]))]
    if folder == "test":
        bproc.writer.write_hdf5(f"{args.output_dir}/{save_path}",data)
        normals = [data["normals"][i] for i in range(len(data["colors"]))]
        depths = [data["depth"][i] for i in range(len(data["depth"]))]
        save_images(normals,environment,args,path_title=save_path,image_type="normal_0000")
        save_images(depths,environment,args,path_title=save_path,image_type="depth_0000",cmap="gray")
    save_images(images,environment,args,path_title=save_path)
    

if not environment['random']:
    generate_dataset()
else:
     test_default_position()


