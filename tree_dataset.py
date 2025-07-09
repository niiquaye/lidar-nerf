import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob

# merges all .xyz files for a tree in a directory

def load_xyz_files_from_directory(directory_path):
    all_points = []
    xyz_files = sorted(glob.glob(os.path.join(directory_path, "*.xyz")))
    for file_path in xyz_files:
        points = np.loadtxt(file_path)
        all_points.append(points)
    if not all_points:
        raise ValueError(f"No .xyz files found in directory: {directory_path}")
    return np.concatenate(all_points, axis=0)

def load_xyz_file(file_path):
    return np.loadtxt(file_path)  # shape: (N, 3)

def get_lidar_rays_from_xyz(points, H=64, W=512, fov_up=30.0, fov_down=-10.0):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d = np.linalg.norm(points, axis=1)

    azimuth = np.arctan2(y, x)  # horizontal angle [-pi, pi]
    elevation = np.arcsin(z / d)  # vertical angle

    fov_total = abs(fov_down) + abs(fov_up)
    h = ((1 - (np.degrees(elevation) + abs(fov_down)) / fov_total) * H).astype(int)
    w = ((0.5 * (azimuth / np.pi + 1.0)) * W).astype(int)

    h = np.clip(h, 0, H - 1)
    w = np.clip(w, 0, W - 1)

    range_image = np.zeros((H, W))
    xyz_image = np.zeros((H, W, 3))
    mask = np.zeros((H, W), dtype=bool)

    for i in range(len(points)):
        if not mask[h[i], w[i]] or d[i] < range_image[h[i], w[i]]:
            range_image[h[i], w[i]] = d[i]
            xyz_image[h[i], w[i]] = points[i]
            mask[h[i], w[i]] = True

    return xyz_image, range_image, mask

class TreePointCloudDataset(Dataset):
    def __init__(self, xyz_path, H=64, W=512):
        self.points = load_xyz_file(xyz_path)
        self.xyz_image, self.range_image, self.mask = get_lidar_rays_from_xyz(
            self.points, H=H, W=W
        )
        self.H, self.W = self.range_image.shape
        self.directions = self.compute_directions()
        self.origins = np.zeros((self.H, self.W, 3))  # Assume LiDAR origin at (0,0,0)
        self.valid_indices = np.argwhere(self.mask)

        self.H_lidar = self.H
        self.W_lidar = self.W

    def compute_directions(self):
        directions = np.zeros((self.H, self.W, 3))
        for h in range(self.H):
            for w in range(self.W):
                point = self.xyz_image[h, w]
                direction = point / (np.linalg.norm(point) + 1e-6)
                directions[h, w] = direction
        return directions

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        h, w = self.valid_indices[idx]
        ray_o = self.origins[h, w]  # Typically [0, 0, 0]
        ray_d = self.directions[h, w]
        point = self.xyz_image[h, w]

        sample = {
            "rays_o_lidar": torch.from_numpy(ray_o).float(),
            "rays_d_lidar": torch.from_numpy(ray_d).float(),
            "images_lidar": torch.from_numpy(point).float()  # Dummy target = 3D point itself
        }
        return sample
