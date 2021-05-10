import bpy
import numpy as np

from src.main.Module import Module
from src.utility.BlenderUtility import check_intersection, check_bb_intersection, duplicate_objects, get_all_blender_mesh_objects
from src.utility.MeshObjectUtility import MeshObject
from src.utility.object.ObjectReplacer import ObjectReplacer


class ObjectReplacerModule(Module):
    """ Replaces mesh objects with another mesh objects and scales them accordingly, the replaced objects and the
        objects to replace with, can be selected over Selectors (getter.Entity).

    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - replace_ratio
          - Ratio of objects in the original scene, which will be replaced. Default: 1.
          - float
        * - copy_properties
          - Copies the custom properties of the objects_to_be_replaced to the objects_to_replace_with. Default:
            True.
          - bool
        * - objects_to_be_replaced
          - Provider (Getter): selects objects, which should be removed from the scene, gets list of objects
            following a certain condition. Default: [].
          - Provider
        * - objects_to_replace_with
          - Provider (Getter): selects objects, which will be tried to be added to the scene, gets list of objects
            following a certain condition. Default: [].
          - Provider
        * - ignore_collision_with
          - Provider (Getter): selects objects, which are not checked for collisions with. Default: [].
          - Provider
        * - max_tries
          - Maximum number of tries to replace one object. Default: 100.
          - int
    """

    def __init__(self, config):
        Module.__init__(self, config)

    def run(self):
        """ Replaces mesh objects with another mesh objects and scales them accordingly, the replaced objects and the objects to replace with in following steps:
        1. Find which object to replace.
        2. Place the new object in place of the object to be replaced and scale accordingly.
        2. If there is no collision, between that object and the objects in the scene, then do replace and delete the original object.

        """
        ObjectReplacer.replace_multiple(
            MeshObject.convert_to_meshes(self.config.get_list("objects_to_be_replaced", [])),
            MeshObject.convert_to_meshes(self.config.get_list("objects_to_replace_with", [])),
            MeshObject.convert_to_meshes(self.config.get_list("ignore_collision_with", [])),
            self.config.get_float("replace_ratio", 1),
            self.config.get_float("copy_properties", 1),
            self.config.get_int("max_tries", 100)
        )
