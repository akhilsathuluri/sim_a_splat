import numpy as np
import click
import pygame
from pathlib import Path
import sys
import viser.transforms as tf
from pydrake.all import Quaternion, RotationMatrix

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
from sim_a_splat.env.manipulator.manipulator_eef_wrapper import ManipulatorEEFWrapper
from sim_a_splat.env.splat.splat_env_wrapper import SplatEnvWrapper


@click.command()
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(render_size, control_hz):
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    package_path = (
        Path(__file__).resolve().parent.parent.parent
        / "sim_a_splat/robot_description/xarm_description/"
    ).__str__()
    package_name = "xarm6/"
    urdf_name = "xarm6_with_push_gripper.urdf"
    eef_link_name = "push_gripper_base_link"
    num_dof = 6
    splat_assets_path = (
        Path(__file__).resolve().parent.parent.parent
        / "sim_a_splat/assets/robots-scene-v2/"
    )
    match_object_name = "xarm6-1"
    splat_config_name = "2024-12-06_150850/config.yml"
    task_assets_path = (
        Path(__file__).resolve().parent.parent.parent
        / "sim_a_splat/assets/tblock_paper/"
    ).__str__()
    task_assets_name = "tblock_paper.obj"
    manipulator_env = ManipulatorSimEnv(
        env_objects=True,
        visualise_flag=True,
        eef_link_name=eef_link_name,
        package_path=package_path,
        package_name=package_name,
        urdf_name=urdf_name,
        num_dof=num_dof,
    )
    manipulator_eef_env = ManipulatorEEFWrapper(manipulator_env)
    camera_setup_info = {
        0: {
            "link_name": "world",
            "local_frame": tf.SE3(
                wxyz_xyz=np.concatenate(
                    (
                        np.array([-0.41946813, 0.89955231, -0.11045113, 0.05150421]),
                        np.array([-0.15, -0.3, -0.05]),
                    )
                )
            ),
            "type": "viewport",
            "render_size": [240, 320],
        },
        1: {
            "link_name": eef_link_name,
            "local_frame": tf.SE3(
                wxyz_xyz=np.concatenate(
                    (np.array([1, 0, 0, 0]), np.array([-0.1, 0, 0.033]))
                )
            ),
            "type": "moving",
            "render_size": [240, 320],
        },
    }
    manipulator_eef_splat_env = SplatEnvWrapper(
        manipulator_eef_env,
        splat_assets_path=splat_assets_path,
        match_object_name=match_object_name,
        splat_config_name=splat_config_name,
        task_assets_path=task_assets_path,
        task_assets_name=task_assets_name,
    )
    manipulator_eef_splat_env._configure_cameras(camera_setup_info)
    obs = manipulator_eef_splat_env.reset(
        reset_to_state={
            "robot_pos": [0.0] * num_dof,
            "block_pos": [0.0, 0.0, 0.0, 0.0],
            "goal_pos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    info_manipulator = manipulator_eef_splat_env.unwrapped._get_info()
    eef_ori = (
        RotationMatrix(Quaternion(info_manipulator["eef_quat"]))
        .ToRollPitchYaw()
        .vector()
    )

    def map_actions(act: np.ndarray):
        if act is None:
            return None
        return np.array([0.25 + 0.45 * act[0] / 298, 0.3 - 0.6 * act[1] / 512, 0.2])

    while True:
        obs = env.reset()
        info = env._get_info()
        _ = env.render(mode="human")

        goal_pose = env.goal_pose
        tblock_pos = info["block_pose"]
        eef_pos = map_actions(info["pos_agent"])
        block_pos = np.array(list(map_actions(tblock_pos[:2])) + [tblock_pos[2]])
        goal_pose = np.array(list(map_actions(goal_pose[:2])) + [goal_pose[2]])
        robot_pos = manipulator_eef_splat_env.env.eefpose2config(
            np.concatenate((eef_pos, eef_ori))
        )
        _ = manipulator_eef_splat_env.reset(
            seed=0,
            reset_to_state={
                "robot_pos": robot_pos,
                "block_pos": block_pos,
                "goal_pos": goal_pose,
            },
        )

        retry = False
        pause = False
        done = False
        terminated = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        while not terminated:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        plan_idx += 1
                        pygame.display.set_caption(f"plan_idx:{plan_idx}")
                        pause = True
                    elif event.key == pygame.K_r:
                        retry = True
                    elif event.key == pygame.K_q:
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            if retry:
                break
            if pause:
                continue

            act = agent.act(obs)
            # _ = splat_env._get_action()
            _, _, _, _ = env.step(act)
            _ = env.render(mode="human")

            # render the splat_env
            if act is not None:
                eef_action = map_actions(act)
                eef_action = {
                    "eef_pos": eef_action,
                    "eef_ori": eef_ori,
                }
                observation, reward, terminated, truncated, info = (
                    manipulator_eef_splat_env.step(eef_action, noobs=True)
                )

            clock.tick(control_hz)


if __name__ == "__main__":
    main()
