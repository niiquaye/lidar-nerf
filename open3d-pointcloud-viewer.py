import numpy as np
import open3d as o3d
import argparse

def load_xyz_file(filepath):
    # Load the .xyz file (expects 3 columns: x, y, z)
    points = np.loadtxt(filepath)
    assert points.shape[1] == 3, "File must contain x, y, z columns only"
    return points

def create_point_cloud(points, color):
    # Create Open3D point cloud object and assign color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def visualize_point_clouds(file1, file2):
    # Load point clouds
    points1 = load_xyz_file(file1)
    points2 = load_xyz_file(file2)

    points2 += np.array([2.0, 0, 0])  # shift 2 units in X direction so clouds dont overlap

    # Assign distinct colors (RGB in [0,1])
    pcd1 = create_point_cloud(points1, color=[1.0, 0.0, 0.0])  # Red
    pcd2 = create_point_cloud(points2, color=[0.0, 0.6, 1.0])  # Light blue

    # Render the point clouds together
    o3d.visualization.draw_geometries([pcd1, pcd2],
                                      window_name="3D Point Cloud Viewer",
                                      point_show_normal=False,
                                      width=1024,
                                      height=768,
                                      mesh_show_back_face=True)


# visualize_point_clouds(file1, file2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render two point clouds from .xyz files")
    parser.add_argument("file1", type=str, help="Path to first .xyz file")
    parser.add_argument("file2", type=str, help="Path to second .xyz file")
    args = parser.parse_args()

    visualize_point_clouds(args.file1, args.file2)
