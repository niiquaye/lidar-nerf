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


# def collate_fn(batch):
#     out = {}
#     for key in batch[0]:
#         out[key] = torch.stack([torch.tensor(b[key]) for b in batch])
#     return out
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)


# ======== CONFIGURATION ========
xyz_path = "data/tree_sample.xyz"  # path to your .xyz file
H, W = 64, 512  # range image resolution - was suggested to pick these dimentsions to go in line with a Velodhyne lidar config
batch_size = 1024
num_epochs = 99
lr = 1e-2

# ======== LOAD DATA ========
dataset = TreePointCloudDataset(xyz_path, H=H, W=W)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# bare bones model
# ======== INITIALIZE MODEL ========
model = NeRFNetwork(
        encoding="hashgrid",
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ======== TRAINING LOOP ========
trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)
trainer.train(num_epochs=num_epochs)

# ======== SAVE MODEL ========
torch.save(model.state_dict(), "lidar_nerf_tree.pth") # will come in handy when rendering point cloud data from model
print("Model saved as lidar_nerf_tree.pth")
