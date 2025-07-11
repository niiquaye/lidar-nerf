import torch
import numpy as np
from lidarnerf.nerf.network import NeRFNetwork
from lidarnerf.nerf.renderer import NeRFRenderer
import os

# Step 1: Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# bare bones model
# ======== INITIALIZE MODEL ========
model = NeRFNetwork(
        encoding="hashgrid",
)

checkpoint = torch.load('/home/niiquaye/lidar-tree-checkpoints/lidar-tree_ep0099.pth', map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval().to(device)

# Step 2: Load your xyz point cloud
points = np.loadtxt("data/tree_sample.xyz")  # shape: (N, 3)
points = torch.from_numpy(points).float().to(device)

# Step 3: Create a 3D grid of points around existing cloud
# For example, jitter original points or sample in bounding box
N = points.shape[0]
factor = 2  # 2x upsampling
jitter = (torch.rand((N * (factor - 1), 3), device=device) - 0.5) * 0.2  # small noise
new_points = points.repeat(factor - 1, 1) + jitter
all_points = torch.cat([points, new_points], dim=0)

# Step 4: Evaluate densities
with torch.no_grad():
    densities = model.density(all_points)["sigma"].squeeze()

# Step 5: Filter points with density above threshold
threshold = 0.00001  # You may adjust this based on your training setup
valid_mask = densities > threshold
upsampled_points = all_points[valid_mask]

# Step 6: Save the new upsampled point cloud
upsampled_np = upsampled_points.detach().cpu().numpy()
np.savetxt("upsampled_tree_data.xyz", upsampled_np)
print(f"Saved upsampled point cloud with {upsampled_np.shape[0]} points.")
