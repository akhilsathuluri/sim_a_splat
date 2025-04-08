# %%
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
from tqdm import tqdm

from sim_a_splat.dataset.splat_pusht_image_dataset import SplatPushTImageDataset
import click

# %%


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, episode_index: int):
        ep_idxs = np.where(dataset.replay_buffer.get_episode_idxs() == episode_index)[0]
        from_idx = ep_idxs[0]
        to_idx = ep_idxs[-1]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert (
        c < h and c < w
    ), f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (
        (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    )
    return hwc_uint8_numpy


# %%

"""
Usage example:
python visualise_dataset.py -e [0,1,2,3] # no space!
"""


@click.command()
@click.option(
    "--episode_indices",
    "-e",
    default="[0, 1]",
    help="List of episode indices to visualize",
)
@click.option("--dataset", "-d", default=None, type=str, help="Path to .zarr dataset")
def main(episode_indices, dataset):
    episode_indices = episode_indices[1:-1]
    episode_indices = episode_indices.split(",")
    episode_indices = np.array(episode_indices, dtype=int)

    data_path = Path("./data/render_splat_pusht_v0.zarr").resolve().__str__()
    dataset = SplatPushTImageDataset(data_path)

    rr.init(f"diffusion_policy/splatpusht/episodes_{episode_indices}", spawn="local")

    for episode_index in episode_indices:
        episode_sampler = EpisodeSampler(dataset, episode_index)
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=8, batch_size=32, sampler=episode_sampler
        )

        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Episode {episode_index}"
        ):
            for ii in range(len(batch["action"])):
                for dim_idx, val in enumerate(batch["action"][ii][0]):
                    rr.log(
                        f"episode_{episode_index}/action/{dim_idx}",
                        rr.Scalar(val.item()),
                    )
                for dim_idx, val in enumerate(batch["obs"]["robot_eef_pos"][ii][0]):
                    rr.log(
                        f"episode_{episode_index}/robot_eef_pos/{dim_idx}",
                        rr.Scalar(val.item()),
                    )
                for jj in range(len(batch["obs"])):
                    if f"camera_{jj}" in batch["obs"]:
                        rr.log(
                            f"episode_{episode_index}/camera_{jj}",
                            rr.Image(
                                to_hwc_uint8_numpy(
                                    batch["obs"][f"camera_{jj}"][ii][0].to(
                                        torch.float32
                                    )
                                )
                            ),
                        )


if __name__ == "__main__":
    main()
