# %%
import torch
import numpy as np
from pathlib import Path
import logging
import viser
import viser.transforms as tf
from sim_a_splat.splat.splat_utils import GSplatLoader
from typing import List
from drake import lcmt_viewer_draw, lcmt_viewer_load_robot
from sim_a_splat.messaging.link import Link
from urchin import URDF
import open3d as o3d
import trimesh
import tempfile
from pydrake.all import RigidTransform

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# %%
class SplatHandler:
    def __init__(
        self,
        splat_assets_path: str,
        match_object_name: str,
        splat_config_name: str,
        package_path: str,
        package_name: str,
        urdf_name: str,
        task_assets_path: str = None,
        task_assets_name: str = None,
        sim_robot_weld_frame_transform: RigidTransform = RigidTransform(),
        server: viser.ViserServer = None,
    ):
        self.instance_uid = match_object_name
        self.sim_robot_weld_frame_transform = sim_robot_weld_frame_transform
        self.links: List[Link] = []
        if server is None:
            self.server = viser.ViserServer()
        else:
            self.server = server
        self.masks_dir = (
            Path(f"{splat_assets_path}/masks/{match_object_name}/").resolve().__str__()
        )
        self._load_saved_masks()
        self._load_saved_splats(
            path=f"{splat_assets_path}/splatfacto/{splat_config_name}"
        )
        self._add_robot_segmented_splats()
        self.robot_description_dir = package_path + "/" + package_name
        self._add_robot_meshes(package_path, package_name, urdf_name)
        self._add_task_meshes(task_assets_path, task_assets_name)
        if server is None:
            self._setup_scene_splat(full_scene=False)

        self.rbt_idx, self.blk_idx = 3, 2
        self.rbt_drake_namespace = f"plant::{urdf_name.split('.')[0]}::"
        self.blk_drake_namespace = f"plant::{task_assets_name.split('.')[0]}::"

    def _load_saved_masks(self):
        link_masks_dict_path = self.masks_dir + "/link_masks_global_dict.npy"
        self.link_masks_dict_saved = np.load(
            link_masks_dict_path, allow_pickle=True
        ).item()
        logging.info(f"Loaded link_masks_dict from {link_masks_dict_path}")
        icp_transformation_path = self.masks_dir + "/icp_transformation.npy"
        icp_transformation = np.load(icp_transformation_path)

        cRmat = icp_transformation[:3, :3]
        cI = cRmat.transpose() @ cRmat
        assert np.all(abs(cI[~np.eye(cI.shape[0], dtype=bool)]) < 1e-6)
        temp_scale = np.mean(cI.diagonal())
        assert np.all(abs(cI.diagonal() - temp_scale) < 1e-6)
        self.scale_factor = np.sqrt(temp_scale)
        Rmat = cRmat / self.scale_factor
        icp_rotation = tf.SO3.from_matrix(Rmat)
        self.icp_transform = tf.SE3.from_rotation_and_translation(
            icp_rotation, icp_transformation[:3, 3]
        )
        self.Ri = self.icp_transform.rotation().as_matrix()
        self.ti = self.icp_transform.translation()

    def _load_saved_splats(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path_to_gsplat = Path(path).resolve()
        bounds = None
        gsplat = GSplatLoader(path_to_gsplat, device)
        if bounds is not None:
            mask = torch.all(
                (gsplat.means - bounds[:, 0] >= 0) & (bounds[:, 1] - gsplat.means >= 0),
                dim=-1,
            )
        else:
            mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

        self.means = gsplat.means[mask].cpu().numpy()
        self.covs = gsplat.covs[mask].cpu().numpy()
        self.colors = gsplat.colors[mask].cpu().numpy()
        self.opacities = gsplat.opacities[mask].cpu().numpy()
        logging.info(f"Loaded splats from {path_to_gsplat}")

    def _setup_scene_splat(self, full_scene=False):
        if full_scene:
            self.server.scene.add_gaussian_splats(
                name="/all",
                centers=self.means,
                covariances=self.covs,
                rgbs=self.colors,
                opacities=self.opacities,
            )
        self.server.scene.add_gaussian_splats(
            name="/scene_ohne_robot",
            centers=self.means[~self.robot_splat_idxs],
            covariances=self.covs[~self.robot_splat_idxs],
            rgbs=self.colors[~self.robot_splat_idxs],
            opacities=self.opacities[~self.robot_splat_idxs],
        )

    def _add_robot_segmented_splats(self):
        self.splat_links_handler = []
        self.robot_splat_idxs = np.zeros(
            next(iter(self.link_masks_dict_saved.values())).shape[0], dtype=bool
        )
        for ii in range(len(self.link_masks_dict_saved.keys())):
            link_name = f"link{ii}"
            idxs = self.link_masks_dict_saved[link_name]
            temp_means_link = self.means[idxs]
            means_link = temp_means_link
            covs_link = self.covs[idxs]
            colors_link = self.colors[idxs]
            opacities_link = self.opacities[idxs]
            splat_link_handler = self.server.scene.add_gaussian_splats(
                name=f"{self.instance_uid}/splat_robot/{link_name}",
                centers=means_link,
                covariances=covs_link,
                rgbs=colors_link,
                opacities=opacities_link,
                wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            )
            self.robot_splat_idxs = np.logical_or(self.robot_splat_idxs, idxs)
            self.splat_links_handler.append(splat_link_handler)

    def _add_robot_meshes(self, package_path=None, package_name=None, urdf_name=None):
        urdf_location = self.robot_description_dir + f"urdf/{urdf_name}"
        with open(urdf_location, "r") as file:
            urdf_content = file.read()
        urdf_content = urdf_content.replace(
            f"package://{package_name}",
            self.robot_description_dir,
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as tmp_urdf_file:
            tmp_urdf_file.write(urdf_content.encode())
            tmp_urdf_location = tmp_urdf_file.name

        robot = URDF.load(tmp_urdf_location)
        joint_config = np.load(self.masks_dir + "/joint_config.npy")
        actuated_joint_names = [
            robot.actuated_joints[ii].name for ii in range(len(robot.actuated_joints))
        ]
        fk = robot.visual_trimesh_fk(cfg=dict(zip(actuated_joint_names, joint_config)))
        translist = list(fk.values())

        meshes = []
        colors = []
        for ii in range(len(robot.links)):
            link = robot.links[ii]
            for visual in link.visuals:
                if visual.geometry.mesh:
                    mesh = o3d.io.read_triangle_mesh(visual.geometry.mesh.filename)
                    meshes.append(mesh)
                    colors.append(visual.material.color)

        self.mesh_trimeshs = []
        self.mesh_frame_handles = []
        self.fk_tf = []
        apply_colors = True
        for ii in range(len(meshes)):
            mesh = meshes[ii]
            mesh_color = np.asarray(mesh.vertex_colors)
            if apply_colors:
                mesh_color = np.tile(colors[ii][:3], (len(mesh.vertices), 1))
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_normals=np.asarray(mesh.vertex_normals),
                vertex_colors=mesh_color,
            )
            mesh_frame_handle = self.server.scene.add_mesh_trimesh(
                name=f"{self.instance_uid}/mesh_robot/link{ii}",
                mesh=mesh_trimesh,
                scale=self.scale_factor,
            )
            self.mesh_trimeshs.append(mesh_trimesh)
            self.mesh_frame_handles.append(mesh_frame_handle)
            self.fk_tf.append(tf.SE3.from_matrix(translist[ii]))

    def _add_task_meshes(self, task_assets_path=None, task_assets_name=None):
        if task_assets_name is not None:
            task_mesh = o3d.io.read_triangle_mesh(
                task_assets_path + "/" + task_assets_name
            )
        else:
            pass
        mesh_color = np.tile(
            np.array([0.956, 0.396, 0.365]), (len(task_mesh.vertices), 1)
        )
        task_mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(task_mesh.vertices),
            faces=np.asarray(task_mesh.triangles),
            vertex_normals=np.asarray(task_mesh.vertex_normals),
            vertex_colors=mesh_color,
        )
        self.task_mesh_frame_handle = self.server.scene.add_mesh_trimesh(
            name=f"{self.instance_uid}/mesh_task/{task_assets_name.split('.')[0]}",
            mesh=task_mesh_trimesh,
            scale=self.scale_factor,
        )

    def load_handler(self, msg: lcmt_viewer_load_robot):
        for lidx in range(msg.num_links):
            link_data = msg.link[lidx]
            link = Link.from_link_data(link_data, name=f"{lidx}")
            self.links.append(link)

    def draw_handler(self, msg: lcmt_viewer_draw):
        local_idx = 0
        weld_transform_translation = self.sim_robot_weld_frame_transform.translation()
        weld_transform_quaternion = (
            self.sim_robot_weld_frame_transform.rotation().ToQuaternion().wxyz()
        )
        assert np.all(
            weld_transform_quaternion == [1.0, 0.0, 0.0, 0.0]
        ), "Weld frame transform quaternions other than identity are not supported currently."

        for idx in range(msg.num_links):
            if msg.robot_num[idx] == self.rbt_idx:
                drake_quaternion = msg.quaternion[idx]
                unit_quaternion = drake_quaternion / np.linalg.norm(drake_quaternion)
                msg_transform_mesh = tf.SE3(
                    wxyz_xyz=np.concatenate(
                        (
                            unit_quaternion,
                            np.array(msg.position[idx] + weld_transform_translation)
                            * self.scale_factor,
                        )
                    )
                )
                msg_transform_splat = tf.SE3(
                    wxyz_xyz=np.concatenate(
                        (
                            unit_quaternion,
                            np.array(msg.position[idx]) + weld_transform_translation,
                        )
                    )
                )
                try:
                    local_msg = self.icp_transform.multiply(msg_transform_mesh)
                    self.mesh_frame_handles[local_idx].wxyz = local_msg.rotation().wxyz
                    self.mesh_frame_handles[local_idx].position = (
                        local_msg.translation()
                    )

                    fk = self.fk_tf[local_idx]
                    Ri = self.Ri
                    ti = self.ti
                    Rfk = fk.rotation().as_matrix()
                    tfk = fk.translation()
                    Rm = msg_transform_splat.rotation().as_matrix()
                    tm = msg_transform_splat.translation()
                    rot = Ri @ Rm @ Rfk.T @ Ri.T
                    pos = (
                        -Ri @ Rm @ Rfk.T @ Ri.T @ ti
                        - self.scale_factor * Ri @ Rm @ Rfk.T @ tfk
                        + self.scale_factor * Ri @ tm
                        + ti
                    )
                    local_msg_splat = tf.SE3.from_rotation_and_translation(
                        tf.SO3.from_matrix(rot), pos
                    )
                    if local_idx < 7:
                        self.splat_links_handler[local_idx].wxyz = (
                            local_msg_splat.rotation().wxyz
                        )
                        self.splat_links_handler[local_idx].position = (
                            local_msg_splat.translation()
                        )
                    local_idx += 1
                except IndexError:
                    logging.warning(
                        f"Warning: Received draw command for non-existent Link index {idx}."
                    )

            # TODO: Similar to the robot idx handling modify objs idx handling
            if msg.robot_num[idx] == self.blk_idx:
                drake_quaternion = msg.quaternion[idx]
                unit_quaternion = drake_quaternion / np.linalg.norm(drake_quaternion)
                msg_transform_mesh = tf.SE3(
                    wxyz_xyz=np.concatenate(
                        (
                            unit_quaternion,
                            np.array(msg.position[idx]) * self.scale_factor,
                        )
                    )
                )
                try:
                    local_msg = self.icp_transform.multiply(msg_transform_mesh)
                    self.task_mesh_frame_handle.wxyz = local_msg.rotation().wxyz
                    self.task_mesh_frame_handle.position = local_msg.translation()
                except IndexError:
                    logging.warning(
                        f"Warning: Received draw command for non-existent Link index {idx}."
                    )

    def get_attached_frame(
        self, body_name: str, local_frame_pos: tf.SE3, msg: lcmt_viewer_draw
    ):
        local_frame_pos = local_frame_pos.translation()
        idx = msg.link_name.index("plant::" + body_name)
        drake_quaternion = msg.quaternion[idx]
        unit_quaternion = drake_quaternion / np.linalg.norm(drake_quaternion)
        msg_transform_frame = tf.SE3(
            wxyz_xyz=np.concatenate(
                (
                    unit_quaternion,
                    np.array(msg.position[idx] + local_frame_pos) * self.scale_factor,
                )
            )
        )
        local_msg = self.icp_transform.multiply(msg_transform_frame)
        return local_msg.rotation().wxyz, local_msg.translation()

    def render(self, chs, cam_poses: List[tf.SE3], render_size: np.ndarray):
        output_imgs = []
        if cam_poses is None:
            cam_poses = [tf.SE3(np.concatenate((chs.wxyz, chs.position)))]
        for ii in range(len(cam_poses)):
            img = chs.get_render(
                height=render_size[ii][0],
                width=render_size[ii][1],
                wxyz=cam_poses[ii].rotation().wxyz,
                position=cam_poses[ii].translation(),
            )
            output_imgs.append(np.array(img))
        return output_imgs


# %%
