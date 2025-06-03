# %%
import torch
from pathlib import Path
from sim_a_splat.splat.splat_utils import GSplatLoader
import viser.transforms as tf
import numpy as np
from gsplat.rendering import rasterization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

means_ex = torch.randn((100, 3), device=device)
quats_ex = torch.randn((100, 4), device=device)
scales_ex = torch.rand((100, 3), device=device) * 0.1
colors_ex = torch.rand((100, 3), device=device)
opacities_ex = torch.rand((100,), device=device)
viewmats = torch.eye(4, device=device)[None, :, :]
Ks = torch.tensor(
    [[300.0, 0.0, 0.0], [0.0, 300.0, 0.0], [0.0, 0.0, 1.0]], device=device
)[None, :, :]
width, height = 300, 200

# %%
# colors, alphas, meta = rasterization(
#     means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, viewmats=viewmats, Ks=Ks, width=width, height=height
# )
# print(colors.shape, alphas.shape)


# %%
path = Path("./assets/scara/splatfacto/2025-04-02_181852/config.yml")
path_to_gsplat = Path(path).resolve()
gsplat = GSplatLoader(path_to_gsplat, device)
mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)

means = gsplat.means[mask]
quats = gsplat.rots[mask]
scales = gsplat.scales[mask]
colors = gsplat.colors[mask]
opacities = gsplat.opacities[mask]
opacities_flat = opacities.view(-1)
cam_pose_01 = tf.SE3(
    wxyz_xyz=np.concatenate(
        (
            # np.array([0.00, 7.818e-01, -6.235e-01, 0.00]),
            # np.array([-0.2, 0.36, 0.01]),
            np.array([0.00, 7.818e-01, -6.235e-01, 0.00]),
            np.array([0.45, -0.15, 0.01]),
        )
    )
)
cam_pose_tensor = torch.tensor(cam_pose_01.as_matrix(), device=device, dtype=torch.float32)[None, :, :]
# cam_pose_tensor = torch.eye(4, device=device)[None, :, :]
width = 320
height = 240
# camera intrinsice
# Ks = torch.tensor(
#     [[300.0, 0.0, 150.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]], device=device
# )[None, :, :]
# for rpi v2 camera
Ks = torch.tensor(
    [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], device=device
)[None, :, :]

# %%
colors, alphas, meta = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities_flat,
    colors=colors,
    viewmats=cam_pose_tensor,
    Ks=Ks,
    width=width,
    height=height,
)

# %%
# displat the rgb image from colors
import matplotlib.pyplot as plt

plt.imshow((colors[0].cpu().numpy()*255).astype(np.uint8))
plt.axis('off')

# %%
# [pypi-options]
# no-build-isolation = ["gsplat"]
# gsplat = { git = "https://github.com/nerfstudio-project/gsplat.git", rev = "v0.1.6" }
