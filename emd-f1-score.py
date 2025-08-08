import numpy as np
import csv
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import argparse
import os

digit = 8

# Load .xyz file (3-column point cloud)
def load_xyz(filename):
    return np.loadtxt(filename)

# Earth Mover's Distance (via optimal bipartite matching)
def compute_emd(pc1, pc2):
    cost_matrix = np.linalg.norm(pc1[:, None, :] - pc2[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

# F-score with distance threshold
def compute_fscore(pc1, pc2, threshold):
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    print(np.linalg.norm(pc1[0] - pc1[1]))  # example intra-cloud distance

    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)

    precision = np.mean(dist1 < threshold)
    recall = np.mean(dist2 < threshold)

    if precision + recall == 0:
        return 0.0, precision, recall

    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall

# Evaluate and save to CSV
def evaluate_point_clouds(file1, file2, output_csv, threshold=0.000000001):
    pc1 = load_xyz(file1)
    pc2 = load_xyz(file2)

    emd = compute_emd(pc1, pc2)
    fscore, precision, recall = compute_fscore(pc1, pc2, threshold)

    # Save results to CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["EMD", f"{emd:.6f}"])
        writer.writerow(["F-score", f"{fscore:.4f}"])
        writer.writerow(["Precision", f"{precision:.4f}"])
        writer.writerow(["Recall", f"{recall:.4f}"])
    
    print(f"Comparison saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two .xyz point clouds and save results to CSV.")
    parser.add_argument("file1", type=str, help="Path to original .xyz file")
    parser.add_argument("file2", type=str, help="Path to compared .xyz file (e.g. upsampled or downsampled)")
    parser.add_argument("-o", "--output", type=str, default=f"comparison_metrics_{digit}.csv", help="Output CSV file")
    parser.add_argument("-t", "--threshold", type=float, default=0.001, help="Distance threshold for F-score")

    args = parser.parse_args()
    evaluate_point_clouds(args.file1, args.file2, args.output, args.threshold)
