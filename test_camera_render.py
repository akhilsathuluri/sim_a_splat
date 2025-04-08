# %%
from sim_a_splat.env.splat.splat_scara_env import SplatEnv
from pathlib import Path
import numpy as np
import imageio
from pydrake.all import RotationMatrix, RollPitchYaw

# %%
package_path = Path(__file__).resolve().parent.parent / "sim_a_splat/robot_description/"
package_name = "scara/"
urdf_name = "scara.urdf"
eef_link_name = "link_3"
splat_env = SplatEnv(
    visualise_sim_flag=True,
    eef_link_name=eef_link_name,
    package_path=package_path,
    package_name=package_name,
    urdf_name=urdf_name,
)
obs = splat_env.reset()
img = splat_env.render()

breakpoint()

# %%

# [imageio.imwrite(f"tmp_{ii}.png", img[ii]) for ii in range(len(img))]
# imageio.imwrite("tmp.png", img)
# breakpoint()

# %%
draw_msg = splat_env._generate_draw_msg()
splat_env.splat_handler.draw_handler(draw_msg)
# fmt: off
moving_camera_poses = splat_env.get_moving_camera_poses( draw_msg, local_frame_pos=np.array([0.05, -0.05, 0.1]))
# fmt: on
splat_env.ch.camera.position = moving_camera_poses[0].translation()
splat_env.ch.camera.wxyz = moving_camera_poses[0].rotation().wxyz
# %% Position 1
# ch = client[0].camera
# ch.position = np.array([-0.1, 0.0, 0.15])
# ch.look_at = np.array([0, 0, -0.1])
# ch.wxyz = RotationMatrix(RollPitchYaw(np.pi, 0, -np.pi / 12)).ToQuaternion().wxyz()

# %%
# position-2
# ch.wxyz = RotationMatrix(RollPitchYaw(-np.pi / 2, 0, 0)).ToQuaternion().wxyz()
# ch.position = np.array([-0.35, 0.05, -0.15])
# ch.look_at = np.array([0, 0, -0.3])

# # %% setting the pose same as the data collected
# ch.position = np.array([-0.35554668,  0.05793625, -0.16483553])
# ch.wxyz = np.array([-0.42796962,  0.60426428, -0.54846474,  0.38844964])

# # %%
# ch.position = np.array([-0.35844819,  0.03081666, -0.14185   ])
# ch.wxyz = np.array([-0.43391441,  0.61863617, -0.53623186,  0.37611562])
# %%
splat_env.ch.camera.position = np.array([-0.2, 0.36, 0.01])
splat_env.ch.camera.wxyz = np.array(
    [2.46086093e-15, 7.81831482e-01, -6.23489802e-01, -2.84992739e-15]
)
