# %%
import sys
from pathlib import Path
import time
import numpy as np
import viser.transforms as tf

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
from sim_a_splat.env.splat.splat_env_wrapper import SplatEnvWrapper


# %%

# package_path = (
#     Path(__file__).resolve().parent.parent.parent
#     / "sim_a_splat/robot_description/xarm_description/"
# ).__str__()
# package_name = "xarm6/"
# urdf_name = "xarm6_with_push_gripper.urdf"
# eef_link_name = "push_gripper_base_link"
# num_dof = 6

# splat_assets_path = (
#     Path(__file__).resolve().parent.parent.parent
#     / "sim_a_splat/assets/robots-scene-v2/"
# )
# match_object_name = "xarm6-1"
# splat_config_name = "2024-12-06_150850/config.yml"
# task_assets_path = (
#     Path(__file__).resolve().parent.parent.parent / "sim_a_splat/assets/tblock_paper/"
# ).__str__()
# task_assets_name = "tblock_paper.obj"


package_path = (
    Path(__file__).resolve().parent.parent.parent / "sim_a_splat/robot_description/"
).__str__()
package_name = "divar113vhw/"
urdf_name = "divar113vhw.urdf"
eef_link_name = "link5"
num_dof = 5

splat_assets_path = (
    Path(__file__).resolve().parent.parent.parent / "sim_a_splat/assets/divar113vhw/"
)
match_object_name = "divar113vhw"
splat_config_name = "2025-06-03_191520/config.yml"
task_assets_path = (
    Path(__file__).resolve().parent.parent.parent / "sim_a_splat/assets/tblock_paper/"
).__str__()
task_assets_name = "tblock_paper.obj"

# TODO: Scara IK fails
# package_path = (
#     Path(__file__).resolve().parent.parent.parent / "sim_a_splat/robot_description/"
# )
# package_name = "scara/"
# urdf_name = "scara.urdf"
# eef_link_name = "gripper"
# num_dof = 3

manipulator_env = ManipulatorSimEnv(
    env_objects=True,
    visualise_flag=True,
    eef_link_name=eef_link_name,
    package_path=package_path,
    package_name=package_name,
    urdf_name=urdf_name,
    num_dof=num_dof,
)

obs = manipulator_env.reset()

# %%

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


# %%

manipulator_splat_env = SplatEnvWrapper(
    manipulator_env,
    splat_assets_path=splat_assets_path,
    match_object_name=match_object_name,
    splat_config_name=splat_config_name,
    task_assets_path=task_assets_path,
    task_assets_name=task_assets_name,
)

manipulator_splat_env._configure_cameras(camera_setup_info)

# %%
obs = manipulator_splat_env.reset()

action = manipulator_splat_env.action_space.sample()
observation, reward, terminated, truncated, info = manipulator_splat_env.step(action)
print(f"Observation: {observation}, Reward: {reward}, Done: {terminated}")
# %% Create a random walk

while True:
    action = obs["robot_joint_pos"] + np.random.uniform(-1, 1, size=num_dof).tolist()
    observation, reward, terminated, truncated, info = manipulator_splat_env.step(
        action
    )
    print(f"Action: {action}, Reward: {reward}, Done: {terminated}")
    time.sleep(0.1)
# %%
