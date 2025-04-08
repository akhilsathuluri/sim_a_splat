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
import secrets

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# %%
class SplatHandler:
    def __init__(
        self,
        splat_dir: str,
        match_object_name: str,
        splat_config_name: str,
        urdf_name: str,
        task_mesh_name: str = None,
        server: viser.ViserServer = None,
    ):
        self.instance_uid = match_object_name
        self.urdf_name = urdf_name
        self.links: List[Link] = []
        if server is None:
            self.server = viser.ViserServer()
        else:
            self.server = server
        self.masks_dir = (
            Path(f"{splat_dir}/masks/{match_object_name}/").resolve().__str__()
        )
        self.robot_mesh_dir = (Path(splat_dir).parent.parent).resolve().__str__()
        self._load_saved_masks()
        self._load_saved_splats(path=f"{splat_dir}/splatfacto/{splat_config_name}")
        self._add_robot_segmented_splats()
        self._add_robot_meshes(self.robot_mesh_dir, self.urdf_name)
        self._add_task_meshes(self.robot_mesh_dir, mesh_name=task_mesh_name)
        if server is None:
            self._setup_scene_splat(full_scene=False)

        self.rbt_idx, self.blk_idx = 3, 2
        self.rbt_drake_namespace = "plant::scara::"
        self.blk_drake_namespace = "plant::scaled_tblock::"

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

    def _add_robot_meshes(self, robot_mesh_dir=None, urdf_name=None):
        robot_model = self.instance_uid.split("-")[0]
        urdf_location = (
            robot_mesh_dir + f"/robot_description/{robot_model}/urdf/{urdf_name}"
        )
        with open(urdf_location, "r") as file:
            urdf_content = file.read()
        urdf_content = urdf_content.replace(
            "package://",
            f"{robot_mesh_dir}/robot_description/",
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".urdf") as tmp_urdf_file:
            tmp_urdf_file.write(urdf_content.encode())
            tmp_urdf_location = tmp_urdf_file.name

        robot = URDF.load(tmp_urdf_location)
        fk = robot.visual_trimesh_fk()
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

    def _add_task_meshes(self, task_mesh_dir=None, mesh_name=None):
        if mesh_name is None:
            tblock_mesh = o3d.io.read_triangle_mesh(
                task_mesh_dir + "/assets/tblock_paper/scaled_tblock.obj"
            )
        else:
            tblock_mesh = o3d.io.read_triangle_mesh(task_mesh_dir + mesh_name)
        mesh_color = np.tile(
            # np.array([76, 160, 224]) / 255, (len(tblock_mesh.vertices), 1)
            np.array([28.6, 47.1, 67.5]) / 255,
            (len(tblock_mesh.vertices), 1),
        )

        tblock_mesh_trimesh = trimesh.Trimesh(
            vertices=np.asarray(tblock_mesh.vertices),
            faces=np.asarray(tblock_mesh.triangles),
            vertex_normals=np.asarray(tblock_mesh.vertex_normals),
            vertex_colors=mesh_color,
        )
        self.tblock_mesh_frame_handle = self.server.scene.add_mesh_trimesh(
            name=f"{self.instance_uid}/mesh_task/tblock",
            mesh=tblock_mesh_trimesh,
            scale=self.scale_factor,
            # scale=1.0,
        )

    def load_handler(self, msg: lcmt_viewer_load_robot):
        for lidx in range(msg.num_links):
            link_data = msg.link[lidx]
            link = Link.from_link_data(link_data, name=f"{lidx}")
            self.links.append(link)

    def draw_handler(self, msg: lcmt_viewer_draw):
        local_idx = 0
        for idx in range(msg.num_links):
            if msg.robot_num[idx] == self.rbt_idx:
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
                msg_transform_splat = tf.SE3(
                    wxyz_xyz=np.concatenate(
                        (
                            unit_quaternion,
                            np.array(msg.position[idx]),
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
                    self.tblock_mesh_frame_handle.wxyz = local_msg.rotation().wxyz
                    self.tblock_mesh_frame_handle.position = local_msg.translation()
                except IndexError:
                    logging.warning(
                        f"Warning: Received draw command for non-existent Link index {idx}."
                    )

    def get_attached_frame(
        self, body_name: str, local_frame_pos: np.ndarray, msg: lcmt_viewer_draw
    ):
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
