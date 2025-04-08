# A simple example to teleop the splat using the xarm in Drake

import numpy as np
import click
from sim_a_splat.common.replay_buffer import ReplayBuffer
from sim_a_splat.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import pygame
from sim_a_splat.env.scara.scara_env import ScaraSimEnv
from pathlib import Path


@click.command()
@click.option("-o", "--output", required=True)
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(output, render_size, control_hz):
    """
    Collect demonstration for the Push-T task.

    Usage: python demo_pusht.py -o data/pusht_demo.zarr

    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area.
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """

    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode="a")

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    # connect to the drake env
    package_path = (
        Path(__file__).resolve().parent.parent / "sim_a_splat/robot_description/"
    )
    package_name = "scara/"
    urdf_name = "scara.urdf"
    eef_link_name = "link_3"

    # connect to the drake env
    drake_instance = ScaraSimEnv(
        env_objects=True,
        visualise_flag=True,
        eef_link_name=eef_link_name,
        package_path=package_path,
        package_name=package_name,
        urdf_name=urdf_name,
    )
    drake_instance.load_model()

    def map_actions(act: np.ndarray):
        if act is None:
            return None
        return np.array([0.075 + 0.3 * act[0] / 298, 0.3 - 0.6 * act[1] / 512, 0.0])

    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f"starting seed {seed}")

        # set seed for env
        env.seed(seed)

        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        goal_pose = env.goal_pose
        info = env._get_info()
        tblock_pos = info["block_pose"]
        # tblock_pose: array([172.        , 122.        ,  -3.28337113])
        # mapped: array([0.50973154, 0.15703125, 0.2       ])

        drake_eef_action = map_actions(info["pos_agent"])
        block_action = np.array(list(map_actions(tblock_pos[:2])) + [tblock_pos[2]])
        goal_pose = np.array(list(map_actions(goal_pose[:2])) + [goal_pose[2]])
        reset_pose = [drake_eef_action, block_action, goal_pose]
        _ = drake_instance.reset(reset_pose)
        dinfo = drake_instance._get_info()

        img = env.render(mode="human")

        # loop state
        retry = False
        pause = False
        done = False
        ddone = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        # step-level while loop
        while not ddone:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f"plan_idx:{plan_idx}")
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry = True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            if retry:
                break
            if pause:
                continue

            act = agent.act(obs)
            dact = drake_instance._get_action()
            if act is not None:
                state = np.concatenate([info["pos_agent"], info["block_pose"]])
                keypoint = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]
                data = {
                    "robot_eef_pos": np.float32(dinfo["eef_pos"]),
                    "action": np.float32(dact),
                    # additional info for rendering
                    "robot_pos": np.float32(dinfo["robot_pos"]),
                    "block_pose": np.float32(dinfo["block_pose"]),
                    "timestamp": np.float32(dinfo["timestamp"]),
                }
                episode.append(data)

            pobs, preward, pdone, pinfo = env.step(act)
            _ = env.render(mode="human")

            if act is not None:
                eef_action = map_actions(act)
                dobs, dreward, ddone, dinfo = drake_instance.step(eef_action)
                drake_instance.render()

            clock.tick(control_hz)

        if not retry:
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack([x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors="disk")
            print(f"saved seed {seed}")
        else:
            print(f"retry seed {seed}")


if __name__ == "__main__":
    main()
