import click
from pathlib import Path

from sim_a_splat.env.splat.splat_env import SplatEnv
from sak.URDFutils import URDFutils
from sak.quickload2drake import robot_joint_teleop
from pydrake.all import *
import time


@click.command()
@click.option("-rs", "--render_size", default=96, type=int)
@click.option("-hz", "--control_hz", default=10, type=int)
def main(render_size, control_hz):
    # meshcat = StartMeshcat()
    # meshcat.Delete()
    # meshcat.DeleteAddedControls()

    package_path = (
        Path(__file__).resolve().parent.parent / "sim_a_splat/robot_description/"
    )
    package_name = "divar113vhw/"
    urdf_name = "divar113vhw.urdf"
    eef_link_name = "link5"
    splat_env = SplatEnv(
        visualise_sim_flag=True,
        eef_link_name=eef_link_name,
        package_path=package_path,
        package_name=package_name,
        urdf_name=urdf_name,
        num_dof=5,
    )
    splat_env.reset()

    urdf_utils = URDFutils(package_path, package_name, urdf_name)
    urdf_utils.modify_meshes(in_mesh_format=".STL")
    urdf_utils.remove_collisions_except([])
    urdf_utils.add_joint_limits()
    urdf_utils.add_actuation_tags()

    # urdf_str, temp_urdf = urdf_utils.get_modified_urdf()
    # robot_joint_teleop(
    #     meshcat=meshcat,
    #     package_path=package_path,
    #     package_name=package_name,
    #     temp_urdf=temp_urdf,
    #     fixed_base=True,
    # )

    # Add 5 sliders to Meshcat
    for i in range(5):
        splat_env.meshcat.AddSlider(
            f"joint_{i}", min=-3.14, max=3.14, step=0.01, value=0.0
        )

    # Read slider values in a loop
    while True:
        joint_values = [
            splat_env.meshcat.GetSliderValue(f"joint_{i}") for i in range(5)
        ]
        print("Joint values:", joint_values)
        time.sleep(0.1)  # Add a small delay to avoid excessive CPU usage
        state = {"robot_pos": joint_values}
        _ = splat_env.set_visual_state(state)


if __name__ == "__main__":
    main()
