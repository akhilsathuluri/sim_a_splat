from pathlib import Path
import time
from timeit import main
import numpy as np
import sys
import viser.transforms as tf
from pydrake.all import RigidTransform, RotationMatrix

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
from sim_a_splat.env.splat.splat_env_wrapper import SplatEnvWrapper
from functools import partial

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except ImportError:
    raise ImportError(
        "rclpy is required for this script to run. Source the ROS 2 environment before running this script."
    )


# create globals for hw interface
global default_joint_state, joint_signs
default_joint_state = np.array([0.92, -3.14, 3.14, 0, 0])
joint_signs = np.array([-1, 1, -1, 1, 1])


def create_splat_env():
    package_path = (
        Path(__file__).resolve().parent.parent.parent / "sim_a_splat/robot_description/"
    ).__str__()
    package_name = "divar113vhw/"
    urdf_name = "divar113vhw.urdf"
    eef_link_name = "link5"
    num_dof = 5
    splat_assets_path = (
        Path(__file__).resolve().parent.parent.parent
        / "sim_a_splat/assets/divar113vhw/"
    )
    match_object_name = "divar113vhw"
    splat_config_name = "2025-06-03_191520/config.yml"
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
        weld_frame_transform=RigidTransform(RotationMatrix(), [0.65, -1.23, 0.42]),
    )

    _ = manipulator_env.reset(
        reset_to_state={
            "robot_pos": [0.0] * num_dof,
            "block_pos": [0.0, 0.0, 0.0, 0.0],
            "goal_pos": [0.0, 0.0, 0.0, 0.0],
        }
    )

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
    manipulator_splat_env = SplatEnvWrapper(
        manipulator_env,
        splat_assets_path=splat_assets_path,
        match_object_name=match_object_name,
        splat_config_name=splat_config_name,
        task_assets_path=task_assets_path,
        task_assets_name=task_assets_name,
    )
    manipulator_splat_env._configure_cameras(camera_setup_info)
    _ = manipulator_splat_env.reset(
        reset_to_state={
            "robot_pos": default_joint_state,
            "block_pos": [0.0, 0.0, 0.0, 0.0],
            "goal_pos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    return manipulator_splat_env


def joint_state_callback(msg, env=None):
    joint_states = np.array(msg.data) * np.pi / 180.0
    joint_states_compensated = joint_states * joint_signs + default_joint_state

    observation, reward, terminated, truncated, info_ = env.step(
        joint_states_compensated, noobs=True
    )


if __name__ == "__main__":
    manipulator_splat_env = create_splat_env()

    rclpy.init(args=None)
    dummy_node = Node("dummy_node")
    dummy_node.get_logger().info("Joint state listener node initialized")

    dummy_node.create_subscription(
        Float32MultiArray,
        "/joint_state",
        partial(joint_state_callback, env=manipulator_splat_env),
        10,
    )

    rclpy.spin(dummy_node)
