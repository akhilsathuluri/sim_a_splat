# %%
import click
import numpy as np
from pathlib import Path
from sim_a_splat.common.replay_buffer import ReplayBuffer
from sim_a_splat.env.splat.splat_scara_env import SplatEnv
from tqdm import tqdm
import os

# %%


@click.command()
@click.option(
    "--dataset_path", "-d", type=str, required=True, help="Path to .zarr dataset"
)
@click.option(
    "--output_path", "-o", type=str, required=True, help="Path to saved rendered data"
)
def main(dataset_path, output_path):
    # load the saved sim data
    zarr_path = Path(dataset_path).resolve().__str__()
    data_keys = ["robot_eef_pos", "action", "robot_pos", "block_pose", "timestamp"]
    sim_data_buffer = ReplayBuffer.copy_from_path(zarr_path, data_keys)
    n_episodes = sim_data_buffer.n_episodes

    # Check output path and determine starting point
    output_path = Path(output_path).resolve().__str__()

    # Create or load existing replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output_path, mode="a")

    # Check if we have already rendered some episodes
    existing_episodes = replay_buffer.n_episodes

    if existing_episodes >= n_episodes:
        print(f"All {n_episodes} episodes have already been rendered. Nothing to do.")
        return

    start_episode = existing_episodes
    print(
        f"Continuing from episode {start_episode}/{n_episodes} ({existing_episodes} already rendered)"
    )

    # initiate the splat env
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

    # Only process episodes that haven't been rendered yet
    for ii in tqdm(range(start_episode, n_episodes), desc="Rendering episodes"):
        _ = splat_env.reset()
        episode_data = sim_data_buffer.get_episode(ii)
        rendered_episode = list()
        for sim_step in tqdm(
            range(len(episode_data["timestamp"])), desc="Rendering steps"
        ):
            step_data = {k: episode_data[k][sim_step] for k in episode_data}
            msg = splat_env.set_visual_state(step_data)
            moving_camera_poses = splat_env.get_moving_camera_poses(
                msg, local_frame_pos=np.array([0.05, -0.05, 0.1])
            )
            _ = splat_env._setup_cameras(
                splat_env.ch, additional_cam_poses=moving_camera_poses
            )
            img_splat = splat_env.render()
            for jj in range(len(img_splat)):
                step_data[f"camera_{jj}"] = img_splat[jj]
            rendered_episode.append(step_data)
        render_data_dict = dict()
        for key in rendered_episode[0].keys():
            render_data_dict[key] = np.stack([x[key] for x in rendered_episode])
        replay_buffer.add_episode(render_data_dict, compressors="disk")
        print(f"Completed episode {ii+1}/{n_episodes}")

    print(f"Rendering complete. All {n_episodes} episodes saved to {output_path}")


if __name__ == "__main__":
    main()
