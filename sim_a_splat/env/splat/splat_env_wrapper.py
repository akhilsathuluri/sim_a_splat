# %%
import gymnasium as gym
import numpy as np
import viser.transforms as tf
import logging
import time
from tqdm import tqdm
from pathlib import Path
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
        # fixed_cam_pose: tf.SE3,
        task_assets_path: Optional[str] = None,
        task_assets_name: Optional[str] = None,
    ):
        super().__init__(env)

        # TODO: has the exact same action space but differs in observation space

        self.ch = self._setup_splats(
            splat_assets_path=splat_assets_path,
            match_object_name=match_object_name,
            splat_config_name=splat_config_name,
            task_assets_path=task_assets_path,
            task_assets_name=task_assets_name,
        )
        # TODO: default fixed cam pose for xarm6-1 robot
        # cam_pose_01 = tf.SE3(
        #     wxyz_xyz=np.concatenate(
        #         (
        #             np.array([-0.41946813, 0.89955231, -0.11045113, 0.05150421]),
        #             np.array([-0.15, -0.3, -0.05]),
        #         )
        #     )
        # )

        # self.fixed_cam_poses = [fixed_cam_pose]
        # self.cam_poses = []

    # frames info: link_name, local_frame, type, render_size
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
        # splat_dir = splat_assets_path + "/assets/robots-scene-v2"
        # match_object_name = "xarm6-1"
        # splat_config_name = "2024-12-06_150850/config.yml"
        # task_mesh_name = "/assets/tblock_paper/tblock_paper.obj"
        self.splat_handler = SplatHandler(
            splat_assets_path,
            match_object_name,
            splat_config_name,
            self.env.package_path,
            self.env.package_name,
            self.env.urdf_name,
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

    # def _setup_cameras(
    #     self, ch, additional_cam_poses=[], view_cam_idx=-1, render_size=[[240, 320]] * 2
    # ):
    #     self.cam_poses = additional_cam_poses + self.fixed_cam_poses
    #     ch.camera.position = self.cam_poses[view_cam_idx].translation()
    #     ch.camera.wxyz = self.cam_poses[view_cam_idx].rotation().wxyz
    #     self.render_size = render_size
    #     return self.cam_poses

    def reset(self, seed: Optional[int] = None, reset_to_state=None):
        self.env.reset(seed=seed, reset_to_state=reset_to_state)
        self.draw_msg = self.env._generate_draw_msg()
        self.splat_handler.draw_handler(self.draw_msg)
        # moving_camera_poses = self.get_moving_camera_poses(draw_msg)
        # _ = self._setup_cameras(self.ch, additional_cam_poses=moving_camera_poses)
        if self.env.visualize_robot_flag:
            self.env.render()
        return self._get_obs()

    # TODO: Check if we still need this method
    def set_visual_state(self, state):
        self.env._set_to_state(state)
        draw_msg = self.env._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        return draw_msg

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
        # wxyz, xyz = self.splat_handler.get_attached_frame(
        #     "push_gripper_base_link", np.array([-0.1, 0, 0.033]), msg
        # )
        # cam_pose_02 = tf.SE3(wxyz_xyz=np.concatenate((wxyz, xyz)))
        # return [cam_pose_02]
        return moving_camera_poses

    def step(self, action, noobs=False):
        # action[2] = 0.2
        obs_in, reward, terminated, truncated, info_in = self.env.step(action)
        draw_msg = self.env._generate_draw_msg()
        self.splat_handler.draw_handler(draw_msg)
        observation = None
        if not noobs:
            # moving_camera_poses = self.get_moving_camera_poses(draw_msg)
            # _ = self._setup_cameras(self.ch, additional_cam_poses=moving_camera_poses)
            if self.env.visualize_robot_flag:
                self.env.render()
            observation = self._get_obs()
        return observation, reward, terminated, truncated, info_in

    def _get_obs(self):
        obs = self.env._get_obs()
        # eef_pos = eef_pos[:2]
        img_out = self.render()
        for ii in range(len(img_out)):
            img_out[ii] = np.moveaxis(img_out[ii], -1, 0)
        # obs = {"robot_eef_pos": eef_pos}
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
        # img_splat = self.splat_handler.render(
        #     chs=self.ch, cam_poses=render_cam_poses, render_size=self.render_size
        # )
        return output_imgs

    def close(self):
        self.splat_handler.server.stop()
        self.env.close()


# %%
