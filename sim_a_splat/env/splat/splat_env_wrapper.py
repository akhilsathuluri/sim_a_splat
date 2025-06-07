# %%
import gymnasium as gym
import numpy as np
import viser.transforms as tf
import logging
import time
from tqdm import tqdm
from sim_a_splat.splat.splat_handler import SplatHandler
from typing import Optional


# %%
class SplatEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        splat_assets_path: str,
        match_object_name: str,
        splat_config_name: str,
        task_assets_path: Optional[str] = None,
        task_assets_name: Optional[str] = None,
    ):
        super().__init__(env)

        self.ch = self._setup_splats(
            splat_assets_path=splat_assets_path,
            match_object_name=match_object_name,
            splat_config_name=splat_config_name,
            task_assets_path=task_assets_path,
            task_assets_name=task_assets_name,
        )

    def _configure_cameras(self, camera_setup_info: dict):
        viewport_camera_info = {
            key: value
            for key, value in camera_setup_info.items()
            if value.get("type") == "viewport"
        }
        self.moving_cameras_info = {
            key: value
            for key, value in camera_setup_info.items()
            if value.get("type") == "moving"
        }
        fixed_cameras_info = {
            key: value
            for key, value in camera_setup_info.items()
            if value.get("type") == "viewport" or value.get("type") == "static"
        }
        self.fixed_cam_poses = []
        for info in fixed_cameras_info.values():
            self.fixed_cam_poses.append(info["local_frame"])

        self.render_cam_keys = list(self.moving_cameras_info.keys()) + list(
            fixed_cameras_info.keys()
        )

        self.ch.camera.position = viewport_camera_info[
            list(viewport_camera_info.keys())[0]
        ]["local_frame"].translation()
        self.ch.camera.wxyz = (
            viewport_camera_info[list(viewport_camera_info.keys())[0]]["local_frame"]
            .rotation()
            .wxyz
        )
        self.camera_setup_info = camera_setup_info

    def _setup_splats(
        self,
        splat_assets_path,
        match_object_name,
        splat_config_name,
        task_assets_path,
        task_assets_name,
        wait_steps=50,
    ):
        self.splat_handler = SplatHandler(
            splat_assets_path,
            match_object_name,
            splat_config_name,
            self.unwrapped.package_path,
            self.unwrapped.package_name,
            self.unwrapped.urdf_name,
            task_assets_path=task_assets_path,
            task_assets_name=task_assets_name,
        )
        for _ in tqdm(range(wait_steps), desc="Waiting for client connection"):
            client = self.splat_handler.server.get_clients()
            if len(client) > 0:
                break
            time.sleep(0.1)
        logging.info(f"client: {client}")
        if len(client) == 0:
            raise RuntimeError("No clients connected after waiting")
        return client[0]

    def reset(self, seed: Optional[int] = None, reset_to_state=None):
        self.unwrapped.reset(seed=seed, reset_to_state=reset_to_state)
        self.draw_msg = self.unwrapped._generate_draw_msg()
        self.splat_handler.draw_handler(self.draw_msg)
        if self.unwrapped.visualize_robot_flag:
            self.env.render()
        return self.unwrapped._get_obs()

    def get_moving_camera_poses(self, msg):
        moving_camera_poses = []
        try:
            for info in self.moving_cameras_info.values():
                link_name = info["link_name"]
                local_frame = info["local_frame"]
                wxyz, xyz = self.splat_handler.get_attached_frame(
                    link_name, local_frame, msg
                )
                moving_camera_poses.append(tf.SE3(wxyz_xyz=np.concatenate((wxyz, xyz))))
        except AttributeError as e:
            logging.error(
                f"Error getting moving camera poses: {e}. Have you configured the cameras with _configure_cameras?"
            )
        return moving_camera_poses

    def step(self, action, noobs=False):
        obs_in, reward, terminated, truncated, info_in = self.env.step(action)
        draw_msg = self.unwrapped._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        observation = None
        if not noobs:
            if self.unwrapped.visualize_robot_flag:
                self.env.render()
            observation = self._get_obs()
        return observation, reward, terminated, truncated, info_in

    def _get_obs(self):
        obs = self.unwrapped._get_obs()
        img_out = self.render()
        for ii in range(len(img_out)):
            img_out[ii] = np.moveaxis(img_out[ii], -1, 0)
        obs.update({f"camera_{ii}": img_out[ii] for ii in range(len(img_out))})
        return obs

    def render(self, mode="rgb_array"):
        self.env.render()
        output_imgs = []

        # TODO: Assumes same order as the render_cam_keys
        moving_camera_poses = self.get_moving_camera_poses(self.draw_msg)
        render_cam_poses = moving_camera_poses + self.fixed_cam_poses
        for ii in range(len(render_cam_poses)):
            img = self.ch.get_render(
                height=self.camera_setup_info[self.render_cam_keys[ii]]["render_size"][
                    0
                ],
                width=self.camera_setup_info[self.render_cam_keys[ii]]["render_size"][
                    1
                ],
                wxyz=render_cam_poses[ii].rotation().wxyz,
                position=render_cam_poses[ii].translation(),
            )
            output_imgs.append(img)
        return output_imgs

    def close(self):
        self.splat_handler.server.stop()
        self.unwrapped.close()


# %%
