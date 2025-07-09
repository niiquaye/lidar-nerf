import torch
from torch.utils.data import DataLoader
from tree_dataset import TreePointCloudDataset
from lidarnerf.nerf.network import NeRFNetwork

from lidarnerf.nerf.utils import (
    seed_everything,
    RMSEMeter,
    MAEMeter,
    DepthMeter,
    PointsMeter,
    Trainer,
)


def lidar_collate_fn(batch):
    rays_o = torch.stack([item["rays_o_lidar"] for item in batch], dim=0)  # [N, 3]
    rays_d = torch.stack([item["rays_d_lidar"] for item in batch], dim=0)  # [N, 3]
    images  = torch.stack([item["images_lidar"] for item in batch], dim=0)  # [N, 3]

    # Add a batch dimension of 1: [1, N, 3]
    return {
        "rays_o_lidar": rays_o.unsqueeze(0),
        "rays_d_lidar": rays_d.unsqueeze(0),
        "images_lidar": images.unsqueeze(0)
    }


from dataclasses import dataclass, field
from typing import List

@dataclass
class Options:
    config: str = "configs/kitti360_1908.txt"
    path: str = "data/kitti360"
    L: bool = False
    test: bool = False
    test_eval: bool = False
    workspace: str = "workspace"
    cluster_summary_path: str = "/summary"
    seed: int = 0
    dataloader: str = "kitti360"
    sequence_id: str = "1908"

    # Lidar-NeRF specific
    enable_lidar: bool = True
    alpha_d: float = 1e3
    alpha_r: float = 1
    alpha_i: float = 1
    alpha_grad_norm: float = 1
    alpha_spatial: float = 0.1
    alpha_tv: float = 1
    alpha_grad: float = 1e2
    intensity_inv_scale: float = 1
    spatial_smooth: bool = False
    grad_norm_smooth: bool = False
    tv_loss: bool = False
    grad_loss: bool = False
    sobel_grad: bool = False

    desired_resolution: int = 2048
    log2_hashmap_size: int = 19
    n_features_per_level: int = 2
    num_layers: int = 2
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    eval_interval: int = 50
    num_rays_lidar: int = 4096
    min_near_lidar: float = 0.01

    depth_loss: str = "l1"
    depth_grad_loss: str = "l1"
    intensity_loss: str = "mse"
    raydrop_loss: str = "mse"

    patch_size_lidar: int = 1
    change_patch_size_lidar: List[int] = field(default_factory=lambda: [1, 1])
    change_patch_size_epoch: int = 2

    # Training options
    iters: int = 30000
    lr: float = 1e-2
    ckpt: str = "latest"
    num_rays: int = 4096
    num_steps: int = 768
    upsample_steps: int = 64
    max_ray_batch: int = 4096
    patch_size: int = 1

    # Network backbone
    fp16: bool = False
    tcnn: bool = False

    # Dataset options
    color_space: str = "srgb"
    preload: bool = False
    bound: float = 2
    scale: float = 0.33
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    dt_gamma: float = 1 / 128
    min_near: float = 0.2
    density_thresh: float = 10
    bg_radius: float = -1


opt = Options()
opt.enable_lidar = True  # Already true by default


# ======== CONFIGURATION ========
xyz_path = "data/tree_sample.xyz"  # path to your .xyz file
H, W = 64, 512  # range image resolution - was suggested to pick these dimentsions to go in line with a Velodhyne lidar config
batch_size = 1024
num_epochs = 99
lr = 1e-2

loss_dict = {
    "mse": torch.nn.MSELoss(reduction="none"),
    "l1": torch.nn.L1Loss(reduction="none"),
    "bce": torch.nn.BCEWithLogitsLoss(reduction="none"),
    "huber": torch.nn.HuberLoss(reduction="none", delta=0.2 * opt.scale),
    "cos": torch.nn.CosineSimilarity(),
}

criterion = {
    "depth": loss_dict[opt.depth_loss],
    "raydrop": loss_dict[opt.raydrop_loss],
    "intensity": loss_dict[opt.intensity_loss],
    "grad": loss_dict[opt.depth_grad_loss],
}

# ======== LOAD DATA ========
dataset = TreePointCloudDataset(xyz_path, H=H, W=W)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lidar_collate_fn)

# bare bones model
# ======== INITIALIZE MODEL ========
model = NeRFNetwork(
        encoding="hashgrid",
)

optimizer_l = lambda model: torch.optim.Adam(
            model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15
        )
# ======== TRAINING LOOP ========
trainer = Trainer(model=model, opt=opt, name='lidar-tree',optimizer=optimizer_l, criterion=criterion)
trainer.train(dataloader, dataloader, num_epochs)


# ======== SAVE MODEL ========
# torch.save(model.state_dict(), "lidar_nerf_tree.pth") # will come in handy when rendering point cloud data from model
# print("Model saved as lidar_nerf_tree.pth")
