# %%
import sys
from pathlib import Path
import time

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv

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

# %%

# obs, info = manipulator_env.reset()
obs, info = manipulator_env.reset(reset_to_state={"robot_pos": [0.0] * 6})

# %%

manipulator_env.step([0.01] * 6)

# %% Create a random walk

while True:
    action = manipulator_env.action_space.sample()
    observation, reward, terminated, truncated, info = manipulator_env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {terminated}")
    time.sleep(0.1)
# %%
