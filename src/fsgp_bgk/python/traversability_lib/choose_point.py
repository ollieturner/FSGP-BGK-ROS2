'''
MIT License

Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def gpu_knn_search(pcl_tensor, k=20):
    distances = torch.cdist(pcl_tensor, pcl_tensor)
    distances, indices = torch.topk(distances, k=k, largest=False)
    return distances, indices

def calculate_curvatures_gpu(pcl_tensor, indices):
    neighbors = pcl_tensor[indices]  
    mean = neighbors.mean(dim=1, keepdim=True)  
    centered = neighbors - mean
    covariance = torch.einsum('bij,bik->bjk', centered, centered) / (neighbors.shape[1] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)  
    curvatures = eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-6)
    return curvatures

def calculate_gradients_gpu(pcl_tensor, indices):
    neighbors = pcl_tensor[indices]  
    dz = neighbors[:, :, 2] - pcl_tensor[:, 2].unsqueeze(1)  
    gradients = torch.mean(torch.abs(dz), dim=1)
    return gradients

def perturb_points_lidar(pcl_tensor, sigma_r, sigma_theta, generator=None):
    # Sensor noise in spherical terms, sensor at the origin of pcl_tensor's frame:
    # range error along the beam u, plus angular error displacing the point by ~r*sigma_theta
    # in the two directions perpendicular to u. Applied once to the whole cloud per sample,
    # before bootstrapping, so a point drawn twice carries the same measurement error.
    r = pcl_tensor.norm(dim=1, keepdim=True).clamp_min(1e-6)
    u = pcl_tensor / r
    ref = torch.zeros_like(u)
    ref[:, 2] = 1.0
    ref[u[:, 2].abs() > 0.999] = torch.tensor([1.0, 0.0, 0.0], device=u.device)
    e1 = torch.linalg.cross(u, ref)
    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp_min(1e-9)
    e2 = torch.linalg.cross(u, e1)

    eps = torch.randn(pcl_tensor.shape[0], 3, generator=generator, device=pcl_tensor.device)
    return pcl_tensor + sigma_r * eps[:, :1] * u + r * sigma_theta * (eps[:, 1:2] * e1 + eps[:, 2:3] * e2)

def bootstrap_features_gpu(pcl_tensor, indices, generator=None):
    # Resample each point's k neighbours with replacement; curvature and gradient are both
    # computed from the same resampled set so their dependence is kept.
    n, k = indices.shape
    draw = torch.randint(0, k, (n, k), generator=generator, device=indices.device)
    boot = indices.gather(1, draw)
    return calculate_curvatures_gpu(pcl_tensor, boot), calculate_gradients_gpu(pcl_tensor, boot)

def extract_features_with_classification_gpu(pcl_arr, curvature_threshold=0.1, gradient_threshold=0.05, voxel_size=0.2, target_num_points=10000, k=20, return_aux=False):
    pcl_positions = pcl_arr[:, :3]
    pcl_intensities = pcl_arr[:, 3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcl_tensor = torch.tensor(pcl_positions, dtype=torch.float32).to(device)

    _, indices = gpu_knn_search(pcl_tensor, k=k)

    curvatures = calculate_curvatures_gpu(pcl_tensor, indices)
    gradients = calculate_gradients_gpu(pcl_tensor, indices)

    is_feature = (curvatures > curvature_threshold) | (gradients > gradient_threshold)
    feature_points = pcl_arr[is_feature.cpu().numpy()]  

    feature_curvatures = curvatures[is_feature.cpu().numpy()].cpu().numpy()
    feature_gradients = gradients[is_feature.cpu().numpy()].cpu().numpy()
    feature_points_with_values = np.hstack((feature_points, feature_curvatures.reshape(-1, 1), feature_gradients.reshape(-1, 1)))

    non_feature_points = pcl_arr[~is_feature.cpu().numpy()]  
    if len(non_feature_points) > 0:
        non_feature_positions = non_feature_points[:, :3]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(non_feature_positions)
        downsampled_pcl = point_cloud.voxel_down_sample(voxel_size)
        downsampled_positions = np.asarray(downsampled_pcl.points)

        downsampled_indices = NearestNeighbors(n_neighbors=1).fit(non_feature_positions).kneighbors(downsampled_positions, return_distance=False)
        average_intensities = np.mean(pcl_intensities[downsampled_indices], axis=1)
        downsampled_points = np.hstack((downsampled_positions, average_intensities.reshape(-1, 1)))

        downsampled_points_extended = np.hstack((downsampled_points, np.zeros((downsampled_points.shape[0], 2))))  
    else:
        downsampled_points_extended = np.empty((0, 6))  

    processed_pcl = np.vstack((feature_points_with_values, downsampled_points_extended))
    # Per output row: index into pcl_arr for feature rows, -1 for downsampled non-feature rows
    # (those keep c = g = 0 and have no source point to recompute features from)
    src_idx = np.concatenate((np.where(is_feature.cpu().numpy())[0], np.full(len(downsampled_points_extended), -1)))

    if len(processed_pcl) > target_num_points:
        choice = np.random.choice(len(processed_pcl), target_num_points, replace=False)
        processed_pcl = processed_pcl[choice]
        src_idx = src_idx[choice]

    if return_aux:
        return processed_pcl, pcl_tensor, indices, src_idx
    return processed_pcl