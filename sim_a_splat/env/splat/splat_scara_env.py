# %%
import numpy as np
import viser.transforms as tf
import logging
import time
from tqdm import tqdm
from pathlib import Path

from sim_a_splat.env.scara.scara_env import ScaraSimEnv
from sim_a_splat.splat.splat_handler import SplatHandler


# %%
class SplatEnv(ScaraSimEnv):
    def __init__(
        self,
        visualise_sim_flag=True,
        eef_link_name=None,
        package_path=None,
        package_name=None,
        urdf_name=None,
    ):
        self.splat_root_dir = (
            Path(__file__).parent.parent.parent.parent.resolve().__str__()
        )
        self.urdf_name = urdf_name
        self.visualise_sim_flag = visualise_sim_flag
        super().__init__(
            visualise_flag=self.visualise_sim_flag,
            eef_link_name=eef_link_name,
            package_path=package_path,
            package_name=package_name,
            urdf_name=urdf_name,
        )
        self.eef_link_name = eef_link_name
        super(SplatEnv, self).load_model()
        self.ch = self._setup_splats()
        cam_pose_01 = tf.SE3(
            wxyz_xyz=np.concatenate(
                (
                    # np.array([-0.41946813, 0.89955231, -0.11045113, 0.05150421]),
                    # np.array([-0.15, -0.3, -0.05]),
                    np.array([0.00, 7.818e-01, -6.235e-01, 0.00]),
                    np.array([-0.2, 0.36, 0.01]),
                )
            )
        )
        self.fixed_cam_poses = [cam_pose_01]
        self.cam_poses = []

    def _setup_splats(self):
        splat_dir = self.splat_root_dir + "/assets/scara"
        match_object_name = "scara"
        splat_config_name = "2025-04-02_181852/config.yml"
        task_mesh_name = "/assets/tblock_paper/scaled_tblock.obj"
        self.splat_handler = SplatHandler(
            splat_dir,
            match_object_name,
            splat_config_name,
            self.urdf_name,
            task_mesh_name=task_mesh_name,
        )
        for _ in tqdm(range(50), desc="Waiting for client connection"):
            client = self.splat_handler.server.get_clients()
            if len(client) > 0:
                break
            time.sleep(0.1)
        logging.info(f"client: {client}")
        if len(client) == 0:
            raise RuntimeError("No clients connected after waiting")
        return client[0]

    def _setup_cameras(
        self, ch, additional_cam_poses=[], view_cam_idx=-1, render_size=[[240, 320]] * 2
    ):
        self.cam_poses = additional_cam_poses + self.fixed_cam_poses
        ch.camera.position = self.cam_poses[view_cam_idx].translation()
        ch.camera.wxyz = self.cam_poses[view_cam_idx].rotation().wxyz
        self.render_size = render_size
        return self.cam_poses

    def reset(self, reset_to_state=None):
        super(SplatEnv, self).reset(reset_to_state)
        draw_msg = super(SplatEnv, self)._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        moving_camera_poses = self.get_moving_camera_poses(draw_msg)
        _ = self._setup_cameras(self.ch, additional_cam_poses=moving_camera_poses)
        if self.visualise_sim_flag:
            super(SplatEnv, self).render()
        return self._get_obs()

    def set_visual_state(self, state):
        super(SplatEnv, self)._set_to_state(state)
        draw_msg = super(SplatEnv, self)._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        return draw_msg

    def get_moving_camera_poses(
        self, msg, local_frame_pos=np.array([0.05, -0.05, 0.1])
    ):
        wxyz, xyz = self.splat_handler.get_attached_frame(
            self.eef_link_name, local_frame_pos, msg
        )
        cam_pose_02 = tf.SE3(wxyz_xyz=np.concatenate((wxyz, xyz)))
        return [cam_pose_02]

    def step(self, action, noobs=False):
        action[2] = 0.2
        _, reward, done, info = super(SplatEnv, self).step(action)
        draw_msg = super(SplatEnv, self)._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        observation = None
        if not noobs:
            moving_camera_poses = self.get_moving_camera_poses(draw_msg)
            _ = self._setup_cameras(self.ch, additional_cam_poses=moving_camera_poses)
            if self.visualise_sim_flag:
                super(SplatEnv, self).render()
            observation = self._get_obs()
        return observation, reward, done, info

    def _get_obs(self):
        eef_pos, _, _, _ = super(SplatEnv, self)._get_obs()
        eef_pos = eef_pos[:2]
        img_out = self.render()
        for ii in range(len(img_out)):
            img_out[ii] = np.moveaxis(img_out[ii], -1, 0)
        obs = {"robot_eef_pos": eef_pos}
        obs.update({f"camera_{ii}": img_out[ii] for ii in range(len(img_out))})
        return obs

    def render(self, mode="rgb_array"):
        super(SplatEnv, self).render()
        img_splat = self.splat_handler.render(
            chs=self.ch, cam_poses=self.cam_poses, render_size=self.render_size
        )
        return img_splat

    def close(self):
        self.splat_handler.server.stop()
        super(SplatEnv, self).close()
