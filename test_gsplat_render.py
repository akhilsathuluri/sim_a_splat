# %%
import torch
from pathlib import Path
from sim_a_splat.splat.splat_utils import GSplatLoader
import viser.transforms as tf
import numpy as np
from gsplat.rendering import rasterization

# %%
path = Path("./assets/scara/splatfacto/2025-04-02_181852/config.yml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            np.array([0.00, 7.818e-01, -6.235e-01, 0.00]),
            np.array([-0.2, 0.36, 0.01]),
        )
    )
)
cam_pose_tensor = torch.tensor(cam_pose_01.as_matrix(), device=device)[None, :, :]
width = 240
height = 320
# camera intrinsice
Ks = torch.tensor(
    [[300.0, 0.0, 150.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]], device=device
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
# [pypi-options]
# no-build-isolation = ["gsplat"]
# gsplat = { git = "https://github.com/nerfstudio-project/gsplat.git", rev = "v0.1.6" }
