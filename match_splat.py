# %%
import open3d as o3d
import numpy as np
import torch
import logging
import tempfile
import viser.transforms as tf
from pathlib import Path
from sim_a_splat.splat.splat_utils import GSplatLoader
from urchin import URDF
from copy import deepcopy as pycopy

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# %%
# --------------------------------------------
# CHANGE TO CUSTOM ROBOT
# urdf_location = (
#     Path("./robot_description/xarm_description/xarm6/urdf/xarm6_with_push_gripper.urdf")
#     .resolve()
#     .__str__()
# )
# # a tag for the object in the splat that is being matched
# match_object_name = "xarm6-2"
# package_tag = "package://xarm6"
# output_dir = (
#     Path("assets/robots-scene-v2/masks" + f"/{match_object_name}/").resolve().__str__()
# )
# splat_path_string = "assets/robots-scene-v2/splatfacto/2024-12-06_150850/config.yml"
# robot_mesh_dir = Path("./robot_description/xarm_description/xarm6/").resolve()

urdf_location = (
    Path("./robot_description/divar113vhw/urdf/divar113vhw.urdf").resolve().__str__()
)
match_object_name = "divar113vhw"
package_tag = "package://divar113vhw"
output_dir = (
    Path("assets/divar113vhw/masks" + f"/{match_object_name}/").resolve().__str__()
)
splat_path_string = "assets/divar113vhw/splatfacto/2025-06-03_191520/config.yml"
robot_mesh_dir = Path("./robot_description/divar113vhw/").resolve()

# --------------------------------------------

# %%
output_dir_path = Path(output_dir)
output_dir_path.mkdir(parents=True, exist_ok=True)
with open(urdf_location, "r") as file:
    urdf_content = file.read()
urdf_content = urdf_content.replace(
    package_tag,
    f"{robot_mesh_dir}",
)
# %%
with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as tmp_urdf_file:
    tmp_urdf_file.write(urdf_content.encode())
    tmp_urdf_location = tmp_urdf_file.name
    print(f"Temporary URDF file created at: {tmp_urdf_location}")
    with open(tmp_urdf_location, "r") as file:
        print(file.read())

# %%
robot = URDF.load(tmp_urdf_location)
# actuated_joints = []
# actuated_joints = robot.actuated_joints
actuated_joint_names = [
    robot.actuated_joints[ii].name for ii in range(len(robot.actuated_joints))
]
# for joint in robot.joints:
#     if joint.joint_type != "fixed":
#         actuated_joints.append(joint.name)
joint_config = np.array([0.2, 3.0, 3.14, 0, 0])
np.save(output_dir + "/joint_config.npy", joint_config)
cfg = dict(zip(actuated_joint_names, joint_config))
translist = robot.visual_geometry_fk(cfg)

# %%
meshes = []
for ii in range(len(robot.links)):
    link = robot.links[ii]
    for visual in link.visuals:
        if visual.geometry:
            mesh = o3d.io.read_triangle_mesh(visual.geometry.mesh.filename)
            transformation = translist[visual.geometry]
            mesh.transform(transformation)
            meshes.append(mesh)

# %%
select_meshes = meshes[:7]
if len(select_meshes) == 1:
    select_meshes = [select_meshes]

# %%

o3d.visualization.draw_plotly(select_meshes)

# %%
combined_mesh = o3d.geometry.TriangleMesh()
for mesh in select_meshes:
    combined_mesh += mesh

# %%
temp_pcd_path = Path(output_dir + "/point_cloud.pcd")
if temp_pcd_path.exists():
    temp_robot_pcd = o3d.io.read_point_cloud(str(temp_pcd_path))
else:
    point_cloud = combined_mesh.sample_points_poisson_disk(number_of_points=20000)
    o3d.io.write_point_cloud(str(temp_pcd_path), point_cloud)
    temp_robot_pcd = point_cloud

# %%
o3d.visualization.draw_plotly([temp_robot_pcd])
robot_pcd = temp_robot_pcd

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_gsplat = (Path(__file__).parent / splat_path_string).resolve()
gsplat = GSplatLoader(path_to_gsplat, device)
rotation = tf.SO3.from_x_radians(0.0).wxyz

# %%
bounds = None
if bounds is not None:
    mask = torch.all(
        (gsplat.means - bounds[0] >= 0) & (bounds[1] - gsplat.means >= 0), dim=-1
    )
else:
    mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

means = gsplat.means[mask].cpu().numpy()
covs = gsplat.covs[mask].cpu().numpy()
colors = gsplat.colors[mask].cpu().numpy()
opacities = gsplat.opacities[mask].cpu().numpy()

splat_pcd = o3d.geometry.PointCloud()
splat_pcd.points = o3d.utility.Vector3dVector(means)

# %%
o3d.visualization.draw_plotly([splat_pcd])

# %%
vol = o3d.visualization.SelectionPolygonVolume()
vol.orthogonal_axis = "Z"

# --------------------------------------------
# CHANGE THESE VALUES TO CROP THE OBJECT TO BE MATCHED
# vol.axis_min = -0.312
# vol.axis_max = 0.2
# # fmt: off
# polygon_bounds = o3d.utility.Vector3dVector([
#     [0.3, -0.06, 0],
#     [0.49, -0.06, 0],
#     [0.49, 0.18, 0],
#     [0.3, 0.18, 0]
# ])
# # fmt: on

vol.axis_min = -0.3
vol.axis_max = 0.1
# fmt: off
polygon_bounds = o3d.utility.Vector3dVector([
    [-0.25, 0.2, 0],
    [0.42, 0.2, 0],
    [0.42, 0.62, 0],
    [-0.25, 0.62, 0]
])
# fmt: on

# --------------------------------------------

vol.bounding_polygon = o3d.utility.Vector3dVector(polygon_bounds)
crop_robot = vol.crop_point_cloud(splat_pcd)
o3d.visualization.draw_plotly([crop_robot])


# %%
np.save(output_dir + "/polygon_bounds.npy", polygon_bounds)
logging.info(f"Saved cropping polygon_bounds to {output_dir + '/polygon_bounds.npy'}")
logging.warning("Using existing bounding polygon guess. Modify if needed.")

# %%
center_crop_robot = crop_robot.get_center()
center_robot_pcd = robot_pcd.get_center()
translate_robot_pcd_to_crop = center_crop_robot - center_robot_pcd
# --------------------------------------------
# CHANGE THESE VALUES TO CUSTOM ROBOT
# R = robot_pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
# threshold = 0.2
# trans_init = np.eye(4)
# scale_init = 1
# adjusted_offset = np.array([0, 0, 0])

R = robot_pcd.get_rotation_matrix_from_xyz((0, 0, 0))
threshold = 0.2
trans_init = np.eye(4)
scale_init = 1
adjusted_offset = np.array([0, 0, 0])
# --------------------------------------------
trans_init[:3, :3] = scale_init * R
trans_init[:3, 3] = translate_robot_pcd_to_crop + adjusted_offset
logging.warning("Using existing transformation guess. Modify if needed.")

# visualisation
temp_robot_pcd = pycopy(robot_pcd)
temp_robot_pcd.translate(trans_init[:3, 3])
temp_robot_pcd.rotate(trans_init[:3, :3])
o3d.visualization.draw_plotly([temp_robot_pcd, crop_robot])

# %%
np.save(output_dir + "/trans_init.npy", trans_init)

estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
estimation_method.with_scaling = True

reg_p2p = o3d.pipelines.registration.registration_icp(
    source=robot_pcd,
    target=crop_robot,
    max_correspondence_distance=threshold,
    init=trans_init,
    estimation_method=estimation_method,
)

# Get the transformation matrix
icp_transformation = reg_p2p.transformation
logging.info("Transformation matrix:")
logging.info(icp_transformation)
np.save(output_dir + "/icp_transformation.npy", icp_transformation)
logging.info(f"Saved icp_transformation to {output_dir}/icp_transformation.npy")
logging.warning(
    "The exported icp_transformation is not an SE3 element! Factor out scaling!"
)

# %%
temp_robot_pcd = pycopy(robot_pcd)
temp_robot_pcd.transform(icp_transformation)
o3d.visualization.draw_plotly([temp_robot_pcd, crop_robot])

# %%
temp_meshes = pycopy(select_meshes)
temp_meshes = [tmesh.transform(icp_transformation) for tmesh in temp_meshes]
o3d.visualization.draw_plotly([temp_robot_pcd, crop_robot] + temp_meshes)

# %%
points = np.asarray(crop_robot.points)
t_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
link_masks_local = []

for tmesh in temp_meshes:
    tmesh_t = o3d.t.geometry.TriangleMesh.from_legacy(tmesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh_t)
    occupancy = scene.compute_occupancy(t_points)
    distances = scene.compute_distance(t_points)
    link_mask = (occupancy.numpy() > 0.5) | (distances.numpy() < 0.015)
    link_masks_local.append(link_mask)

colored_points = np.zeros((points.shape[0], 3))

colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
]

for i, link_mask in enumerate(link_masks_local):
    colored_points[link_mask] = colors[i % len(colors)]

colored_pcd = o3d.geometry.PointCloud()
colored_pcd.points = o3d.utility.Vector3dVector(points)
colored_pcd.colors = o3d.utility.Vector3dVector(colored_points)

o3d.visualization.draw_plotly([colored_pcd])

# %%

link_masks_global = [
    np.isin(
        means, np.array(crop_robot.points)[link_masks_local[ii]], assume_unique=True
    ).all(axis=1)
    for ii in range(len(link_masks_local))
]
link_masks_dict = {
    f"link{i}": link_masks_global[i] for i in range(len(link_masks_local))
}
np.save(output_dir + "/link_masks_global_dict.npy", link_masks_dict)
logging.info(f"Saved link_masks_global_dict to {output_dir}\link_masks_global_dict.npy")


# %% Testing

# Load the link masks dictionary
link_masks_dict_path = Path(output_dir + "/link_masks_global_dict.npy")
test_link_masks_dict_saved = np.load(link_masks_dict_path, allow_pickle=True).item()
icp_transformation_path = Path(output_dir + "/icp_transformation.npy")
test_icp_transformation = np.load(icp_transformation_path)

rmat = test_icp_transformation[:3, :3]
rmat.transpose() @ rmat
np.round(rmat.transpose() @ rmat, decimals=6)
print(f"Scaling is: {(np.round(rmat.transpose() @ rmat, decimals=6)).trace() / 3}")

assert ((np.round(rmat.transpose() @ rmat, decimals=6)).trace() / 3) == (
    np.round(rmat.transpose() @ rmat, decimals=6)
).diagonal()[0]

# %%
