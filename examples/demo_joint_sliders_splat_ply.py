from pathlib import Path
import logging
import time
import numpy as np
import sys
import viser.transforms as tf

logging.basicConfig(level=logging.WARNING)

sys.path.append(Path(__file__).resolve().parent.parent.__str__())
from sim_a_splat.env.manipulator.manipulator_env import ManipulatorSimEnv
from sim_a_splat.env.splat.ply_splat_env_wrapper import PlySplatEnvWrapper


def main():
    root = Path(__file__).resolve().parent.parent

    # --------------------------------------------
    # CHANGE TO CUSTOM ROBOT
    package_path = (root / "robot_description").__str__()
    package_name = "franka_dual_arm/"
    urdf_name = "fr3_duo_drake.urdf"
    eef_link_name = ["right_fr3v2_link8", "left_fr3v2_link8"]  # check URDF link names
    num_dof = 18  # 7 arm + 2 finger joints per arm × 2 arms

    splat_assets_path = root / "assets/cppoc"
    match_object_name = "fr3_duo_drake"
    ply_name = "export_30000_cropped.ply"

    manipulator_env = ManipulatorSimEnv(
        env_objects=False,
        visualise_flag=True,
        eef_link_name=eef_link_name,
        package_path=package_path,
        package_name=package_name,
        urdf_name=urdf_name,
        num_dof=num_dof,
    )

    # Load initial config from joint_config.npy and map to actual joint names
    joint_config_dict = np.load(
        splat_assets_path / "masks" / match_object_name / "joint_config.npy",
        allow_pickle=True,
    ).item()

    # Get actuated joint names from the drake plant (scoped to robot model)
    plant = manipulator_env.plant
    model = manipulator_env.robot_model_instance
    joint_names = plant.GetPositionNames(model, always_add_suffix=False)

    # Extract values from dict using actual joint names, default to 0 if not found
    home_config = np.array([joint_config_dict.get(name, 0.0) for name in joint_names])

    splat_assets_path = splat_assets_path.__str__()
    # --------------------------------------------

    camera_setup_info = {
        "viewport": {
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
    }

    manipulator_splat_env = PlySplatEnvWrapper(
        manipulator_env,
        splat_assets_path=splat_assets_path,
        match_object_name=match_object_name,
        ply_name=ply_name,
    )
    manipulator_splat_env._configure_cameras(camera_setup_info)
    _ = manipulator_splat_env.reset(
        reset_to_state={
            "robot_pos": home_config,
        }
    )

    for i in range(num_dof):
        manipulator_splat_env.unwrapped.meshcat.AddSlider(
            f"joint_{i}", min=-3.14, max=3.14, step=0.01, value=home_config[i]
        )

    while True:
        joint_values = np.array(
            [
                manipulator_splat_env.unwrapped.meshcat.GetSliderValue(f"joint_{i}")
                for i in range(num_dof)
            ]
        )
        print("Joint values:", joint_values)
        observation, reward, terminated, truncated, info_ = manipulator_splat_env.step(
            joint_values, noobs=True
        )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
