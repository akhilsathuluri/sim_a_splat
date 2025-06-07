# %%
import sys
from pathlib import Path
import time
from pydrake.all import Quaternion, RotationMatrix, RollPitchYaw

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
from sim_a_splat.env.manipulator.manipulator_eef_wrapper import ManipulatorEEFWrapper
import numpy as np

package_path = (
    Path(__file__).resolve().parent.parent.parent
    / "sim_a_splat/robot_description/xarm_description/"
)
package_name = "xarm6/"
urdf_name = "xarm6_with_push_gripper.urdf"
eef_link_name = "push_gripper_base_link"
num_dof = 6

# package_path = (
#     Path(__file__).resolve().parent.parent.parent / "sim_a_splat/robot_description/"
# )
# package_name = "divar113vhw/"
# urdf_name = "divar113vhw.urdf"
# eef_link_name = "link5"
# num_dof = 5

# TODO: Scara IK fails
# package_path = (
#     Path(__file__).resolve().parent.parent.parent / "sim_a_splat/robot_description/"
# )
# package_name = "scara/"
# urdf_name = "scara.urdf"
# eef_link_name = "gripper"
# num_dof = 3

manipulator_env = ManipulatorSimEnv(
    env_objects=False,
    visualise_flag=True,
    eef_link_name=eef_link_name,
    package_path=package_path,
    package_name=package_name,
    urdf_name=urdf_name,
    num_dof=num_dof,
)

obs = manipulator_env.reset(
    reset_to_state={
        "robot_pos": [0.0] * num_dof,
        "block_pos": [0.0, 0.0, 0.0, 0.0],
        "goal_pos": [0.0, 0.0, 0.0, 0.0],
    }
)
info = manipulator_env._get_info()

# %%
manipulator_eef_env = ManipulatorEEFWrapper(manipulator_env)
eef_pos_init = info["eef_pos"]
eef_quat_init = info["eef_quat"]
print(f"Initial EEF Position: {eef_pos_init}, Initial EEF Quaternion: {eef_quat_init}")
# %%

# eef_action_sample = manipulator_eef_env.action_space.sample()
eef_action_sample = {
    "eef_pos": eef_pos_init + np.array([0.0, 0.0, 0.0]),
    "eef_ori": RotationMatrix(Quaternion(eef_quat_init)).ToRollPitchYaw().vector(),
}
observation, reward, terminated, truncated, info = manipulator_eef_env.step(
    eef_action_sample
)
print(f"Observation: {observation}, Info: {info}")

# %% Create a random walk

while True:
    # action = eef_pos_init
    action = eef_pos_init + np.random.uniform(-0.01, 0.01, size=3).tolist()
    # action = eef_pos_init + [0, np.random.uniform(-0.01, 0.01, size=1)[0], 0]
    # action = eef_pos_init + [np.random.uniform(-0.000, 0.000, size=1)[0], 0, 0]
    action = {
        "eef_pos": action,
        "eef_ori": RotationMatrix(Quaternion(eef_quat_init)).ToRollPitchYaw().vector(),
    }
    observation, reward, terminated, truncated, info = manipulator_eef_env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {terminated}")
    time.sleep(0.1)
# %%
