import numpy as np
import click
import pygame
from pathlib import Path
from sim_a_splat.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

from sim_a_splat.env.splat.splat_scara_env import SplatEnv

# from sim_a_splat.env.scara.scara_env import ScaraSimEnv


@click.command()
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(render_size, control_hz):
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
    agent = env.teleop_agent()
    clock = pygame.time.Clock()

    package_path = (
        Path(__file__).resolve().parent.parent / "sim_a_splat/robot_description/"
    )
    package_name = "scara/"
    urdf_name = "scara.urdf"
    eef_link_name = "link_3"

    # connect to the drake env
    splat_env = SplatEnv(
        visualise_sim_flag=True,
        eef_link_name=eef_link_name,
        package_path=package_path,
        package_name=package_name,
        urdf_name=urdf_name,
    )
    splat_env.load_model()

    def map_actions(act: np.ndarray):
        if act is None:
            return None
        return np.array([0.075 + 0.3 * act[0] / 298, 0.3 - 0.6 * act[1] / 512, 0.0])

    while True:
        obs = env.reset()
        info = env._get_info()
        _ = env.render(mode="human")

        goal_pose = env.goal_pose
        tblock_pos = info["block_pose"]
        eef_pos = map_actions(info["pos_agent"])
        block_pos = np.array(list(map_actions(tblock_pos[:2])) + [tblock_pos[2]])
        goal_pose = np.array(list(map_actions(goal_pose[:2])) + [goal_pose[2]])
        reset_pose = [eef_pos, block_pos, goal_pose]
        _ = splat_env.reset(reset_pose)

        retry = False
        pause = False
        done = False
        sdone = False
        plan_idx = 0
        pygame.display.set_caption(f"plan_idx:{plan_idx}")
        while not sdone:
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
            dact = splat_env._get_action()
            print(f"act: {act}, dact: {dact}")
            _, _, _, _ = env.step(act)
            _ = env.render(mode="human")

            if act is not None:
                eef_action = map_actions(act)
                _, _, sdone, _ = splat_env.step(eef_action, noobs=True)

            clock.tick(control_hz)


if __name__ == "__main__":
    main()
