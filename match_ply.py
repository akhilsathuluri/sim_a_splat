# %%
import open3d as o3d
import numpy as np
import logging
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from plyfile import PlyData
from urchin import URDF
from copy import deepcopy as pycopy

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

_C0 = 0.28209479177387814


def sh2rgb(sh):
    return sh * _C0 + 0.5


# %%
# --------------------------------------------
# CHANGE TO CUSTOM ROBOT
urdf_location = (
    Path("./robot_description/franka_dual_arm/urdf/fr3_duo_drake.urdf")
    .resolve()
    .__str__()
)
match_object_name = "fr3_duo_drake"
package_tag = "package://franka_dual_arm"
output_dir = Path("assets/cppoc/masks" + f"/{match_object_name}/").resolve().__str__()
ply_path_string = "assets/cppoc/export_30000_cropped.ply"
robot_mesh_dir = Path("./robot_description/franka_dual_arm/").resolve()


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
actuated_joint_names = [
    robot.actuated_joints[ii].name for ii in range(len(robot.actuated_joints))
]
# get sorted names
sorted_joint_names = sorted(actuated_joint_names)
right_robot_home = [0.0, 0.0, 0.0, 0.0, -2.0, 0.9, 1.7, 0.0]
left_robot_home = [0.0, 0.0, 0.0, 0.0, -2.0, -0.9, 1.7, -np.pi / 2]

joint_config = np.array(right_robot_home + left_robot_home)
cfg = dict(zip(sorted_joint_names, joint_config))
np.save(output_dir + "/joint_config.npy", cfg, allow_pickle=True)
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
select_meshes = meshes
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

path_to_ply = (Path(__file__).parent / ply_path_string).resolve()
plydata = PlyData.read(str(path_to_ply))
v = plydata.elements[0]

# %%
bounds = None
means = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
colors = sh2rgb(
    np.stack(
        [np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])],
        axis=1,
    )
)

if bounds is not None:
    mask = np.all((means - bounds[0] >= 0) & (bounds[1] - means >= 0), axis=-1)
else:
    mask = np.ones(means.shape[0], dtype=bool)

splat_pcd = o3d.geometry.PointCloud()
splat_pcd.points = o3d.utility.Vector3dVector(means)
splat_pcd.colors = o3d.utility.Vector3dVector(colors)

# %%
o3d.visualization.draw_plotly([splat_pcd])

# %%
# --------------------------------------------
# CHANGE THESE VALUES TO CROP THE OBJECT TO BE MATCHED
x_min, x_max = -0.405, -0.01
y_min, y_max = -0.38, 0.0
z_min, z_max = -0.30, 0.30
# --------------------------------------------

polygon_bounds = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])

bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[x_min, y_min, z_min],
    max_bound=[x_max, y_max, z_max],
)
crop_robot = splat_pcd.crop(bbox)

splat_pcd_context = pycopy(splat_pcd)
splat_pcd_context.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_plotly([splat_pcd_context, crop_robot])

# %%
np.save(output_dir + "/polygon_bounds.npy", polygon_bounds)
logging.info(f"Saved cropping polygon_bounds to {output_dir + '/polygon_bounds.npy'}")
logging.warning("Using existing bounding polygon guess. Modify if needed.")

# %%
trans_init = np.eye(4)
scale_init = 0.5
adjusted_offset = np.array([0.4, 0.2, 0.32])
R = robot_pcd.get_rotation_matrix_from_xyz((-np.pi / 2, np.pi, 0))

trans_init[:3, 3] = crop_robot.get_center() - robot_pcd.get_center() + adjusted_offset
trans_init[:3, :3] = scale_init * R
logging.warning("Using existing transformation guess. Modify if needed.")

# visualisation
temp_robot_pcd = pycopy(robot_pcd)
temp_robot_pcd.transform(trans_init)
o3d.visualization.draw_plotly([temp_robot_pcd, crop_robot])

# %%
np.save(output_dir + "/trans_init.npy", trans_init)

estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
estimation_method.with_scaling = True

threshold = 0.22
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

n_links = len(link_masks_local)
link_colors = np.array(
    [plt.get_cmap("tab20")(i / max(n_links, 1))[:3] for i in range(n_links)]
)

for i, link_mask in enumerate(link_masks_local):
    colored_points[link_mask] = link_colors[i]

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
