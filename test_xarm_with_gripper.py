# %%

import numpy as np
from pathlib import Path
import pydrake.multibody.plant as pmp
from pydrake.all import *
from sak.URDFutils import URDFutils

from sak.quickload2drake import robot_joint_teleop

# %%

meshcat = StartMeshcat()
meshcat.Delete()
meshcat.DeleteAddedControls()

# %%
package_path = (
    Path(__file__).resolve().parent.parent
    / "sim_a_splat/robot_description/xarm_description"
)
package_name = "xarm6/"
# urdf_name = "xarm6_with_gripper.urdf"
urdf_name = "xarm6_with_parallel_gripper.urdf"

urdf_utils = URDFutils(package_path, package_name, urdf_name)
urdf_utils.modify_meshes()
urdf_utils.remove_collisions_except([])
urdf_utils.add_actuation_tags()

urdf_str, temp_urdf = urdf_utils.get_modified_urdf()

# %%

# builder = DiagramBuilder()
# plant, scene_graph = pmp.AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
# parser = Parser(plant, scene_graph)
# abs_path = Path(package_path).resolve().__str__()
# parser.package_map().Add(package_name.split("/")[0], abs_path + "/" + package_name)
# print(package_name.split("/")[0])
# model = parser.AddModels(temp_urdf.name)[0]
# plant.Finalize()

# left_finger_frame = plant.GetFrameByName("right_finger")
# base_link_frame = plant.GetFrameByName("xarm_gripper_base_link")
# context = plant.CreateDefaultContext()
# X_left_finger_in_base = plant.CalcRelativeTransform(
#     context, base_link_frame, left_finger_frame
# )


# %%
robot_joint_teleop(
    meshcat=meshcat,
    package_path=package_path,
    package_name=package_name,
    temp_urdf=temp_urdf,
)

# %%
