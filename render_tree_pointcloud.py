import torch
import numpy as np
from tree_dataset import get_lidar_rays_from_xyz
from lidarnerf.nerf.network import NeRFNetwork
import os

def forward_rays(model, rays_o, rays_d, step_size=0.01, max_dist=20.0):
    N = rays_o.shape[0]
    device = rays_o.device
    t_vals = torch.arange(0.0, max_dist, step_size).to(device)  # shape: (T,)
    t_vals = t_vals.expand(N, -1)  # (N, T)

    pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)  # (N, T, 3)
    dirs = rays_d.unsqueeze(1).expand_as(pts)  # (N, T, 3)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    raw = model(pts_flat, dirs_flat)
    sigma = raw['sigma'].view(N, -1)
    weights = 1.0 - torch.exp(-sigma * step_size)  # (N, T)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-5)
    est_depth = (t_vals * weights).sum(dim=1)

    return est_depth

def compute_ray_directions(H, W, fov_up=30.0, fov_down=-10.0):
    fov_total = abs(fov_down) + abs(fov_up)
    directions = np.zeros((H, W, 3), dtype=np.float32)

    for h in range(H):
        for w in range(W):
            elev = fov_up - (fov_total * h / H)
            azim = 360.0 * w / W - 180.0

            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)

            x = np.cos(elev_rad) * np.cos(azim_rad)
            y = np.cos(elev_rad) * np.sin(azim_rad)
            z = np.sin(elev_rad)
            directions[h, w] = [x, y, z]

    return directions

def render_dense_point_cloud(model_path, base_xyz_file, H=128, W=1024, output_path="dense_output.xyz"):
    base_points = np.loadtxt(base_xyz_file)
    _, _, mask = get_lidar_rays_from_xyz(base_points, H=H, W=W)
    directions = compute_ray_directions(H, W)

    ray_dirs = directions[mask].reshape(-1, 3)
    ray_origins = np.zeros_like(ray_dirs, dtype=np.float32)

    model = NeRFNetwork(
        encoding="hashgrid",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        rays_o = torch.from_numpy(ray_origins).float().cuda()
        rays_d = torch.from_numpy(ray_dirs).float().cuda()

        distances = forward_rays(model, rays_o, rays_d).cpu().numpy()

    upsampled_xyz = ray_dirs * distances[:, np.newaxis]
    np.savetxt(output_path, upsampled_xyz, fmt="%.6f")
    print(f"Saved upsampled point cloud with {upsampled_xyz.shape[0]} points to: {output_path}")

if __name__ == "__main__":
    render_dense_point_cloud(
        model_path="lidar_nerf_tree.pth",
        base_xyz_file="data/tree_sample.xyz",
        H=128,
        W=1024,
        output_path="dense_output.xyz"
    )
