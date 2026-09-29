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
from sklearn.neighbors import NearestNeighbors
import traversability_lib.choose_point as choose_point
from sklearn.decomposition import PCA
import torch
import gpytorch
from traversability_lib.sgp_model import SGPModel
from traversability_lib.traversability import TraversabilityAnalyzerWithBGK_GPU
import yaml
from traversability_lib import point_cloud_tool
import time

class TraversabilityAnalyzer:
    def __init__(self, config_path="../config/params.yaml"):
        self.load_config(config_path)
        self.initialize_components()

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.curvature_threshold = config["curvature_threshold"]
        self.gradient_threshold = config["gradient_threshold"]
        self.key_voxel_size = config["key_voxel_size"]
        self.inducing_points = config["inducing_points"]
        self.lengthscale = config["lengthscale"]
        self.alpha = config["alpha"]
        self.apply_kernel_init = config["apply_kernel_init"]
        self.gp_train_iters = config["gp_train_iters"]
        self.gp_lr = config["gp_lr"]
        self.gp_noise_init = config["gp_noise_init"]

        self.resolution = config["resolution"]
        self.x_length = config["x_length"]
        self.y_length = config["y_length"]

        self.max_slope = config["max_slope"]
        self.min_slope = config["min_slope"]
        self.test_slope = config["test_slope"]

        self.max_flatness = config["max_flatness"]
        self.min_flatness = config["min_flatness"]
        self.test_flatness = config["test_flatness"]

        self.max_height = config["max_height"]
        self.min_height = config["min_height"]
        self.test_height = config["test_height"]

        self.max_uncertainty = config["max_uncertainty"]
        self.min_uncertainty = config["min_uncertainty"]
        self.test_uncertainty = config["test_uncertainty"]

        self.w_slope = config["w_slope"]
        self.w_flatness = config["w_flatness"]
        self.w_step_height = config["w_step_height"]
        
        self.time_window = config["time_window"]
        self.max_history_frames = config["max_history_frames"]
        
        self.i_num = config["i_num"]
        self.base_height = config["base_height"]
        self.bgk_threshold = config["bgk_threshold"]
        self.downsampl_voxel_size = config["downsampl_voxel_size"]
        self.open_pca=config["open_pca"]

        self.num_samples = config["num_samples"]
        self.knn_k = config["knn_k"]
        self.lidar_range_std = config["lidar_range_std"]
        self.lidar_angular_std = np.deg2rad(config["lidar_angular_std_deg"])
        self.sample_seed = config["sample_seed"]
        self.grad_cov_jitter = config["grad_cov_jitter"]
        self.smooth_sample_slope = config["smooth_sample_slope"]
        if self.num_samples > 0 and self.open_pca:
            # PCA mixes the GP inputs, so columns 0/1 would no longer be x/y for the spatial gradient
            raise ValueError("num_samples > 0 requires open_pca: False")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_generator = torch.Generator(device=self.device)
        if self.sample_seed >= 0:
            self.sample_generator.manual_seed(self.sample_seed)
        else:
            self.sample_generator.seed()

    def initialize_components(self):
        self.pca = PCA(n_components=4)
        self.grid = None
        self.Xs = None
        self.Ys = None
        self.Zs = None
        self.curvatures = None
        self.gradients = None

        self.analyzer = TraversabilityAnalyzerWithBGK_GPU(
            self.resolution, self.x_length, self.y_length, self.time_window, self.max_history_frames
        )
        
        self.mean = None
        self.var = None
        self.grad_mean = None
        self.slope = None
        self.flatness = None
        self.step_height = None
        self.uncertainty = None
        self.traversability = None
        self.traversability_samples = None

        self.pose = None
        self.traversability_dict = None
        self.kd_tree = None
        self.data_dict = None

    def fake_intensity(self, pcl, num_points):
        num_original_points = pcl.shape[0]
        return np.hstack([pcl, np.ones((num_original_points, 1))])

    def _fit_grid_interp(self):
        # Grid cells and their fixed k=5 inverse-distance weights to the key points; computed
        # once per frame and reused by every sample, since key point positions don't change.
        x_range = self.x_length / 2
        y_range = self.y_length / 2
        x_s = np.arange(-x_range, x_range, self.resolution, dtype='float32')
        y_s = np.arange(-y_range, y_range, self.resolution, dtype='float32')

        self.grid_xy = np.array(np.meshgrid(x_s, y_s)).T.reshape(-1, 2)
        X_train = np.column_stack((self.Xs, self.Ys))

        knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_train)
        distances, self.grid_knn_idx = knn.kneighbors(self.grid_xy)

        weights = 1 / (distances + 1e-6)
        self.grid_knn_w = weights / np.sum(weights, axis=1, keepdims=True)

    def _interp_features(self, curvatures_train, gradients_train):
        curvatures_pred = np.sum(self.grid_knn_w[:, :, None] * curvatures_train[self.grid_knn_idx], axis=1)
        gradients_pred = np.sum(self.grid_knn_w[:, :, None] * gradients_train[self.grid_knn_idx], axis=1)
        return curvatures_pred, gradients_pred

    def sampling_grid(self):
        self._fit_grid_interp()
        curvatures_pred, gradients_pred = self._interp_features(self.curvatures, self.gradients)
        self.grid = np.column_stack((self.grid_xy, curvatures_pred, gradients_pred))

    def filter_low_uncertainty_data(self, mean, var, grad_mean, grid, threshold=1.35):
        low_uncertainty_indices = np.where(var < threshold)[0]
        self.keep_idx = low_uncertainty_indices
        filtered_mean = mean[low_uncertainty_indices]
        filtered_var = var[low_uncertainty_indices]
        filtered_grad_mean = grad_mean[low_uncertainty_indices]
        filtered_grid = grid[low_uncertainty_indices]
        
        return filtered_mean, filtered_var, filtered_grad_mean, filtered_grid

    def generate_robot_points(self, l, w):
        x = np.linspace(-l/2, l/2, num=self.i_num)
        y = np.linspace(-w/2, w/2, num=self.i_num)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.full_like(x, self.base_height)
        i = np.full_like(x, 0)
        c = np.full_like(x, 0)
        g = np.full_like(x, 0)
        
        return np.column_stack((x, y, z, i, c, g))

    def generate_local_traversability_map(self, pose, transformed_points):
        self.pose = pose
        start_time = time.time()
        expanded_pcl = self.fake_intensity(transformed_points, 5000)
        expanded_pcl = point_cloud_tool.voxel_downsample(expanded_pcl, voxel_size=self.downsampl_voxel_size)

        key_points, self.pcl_tensor, self.knn_indices, key_src_idx = choose_point.extract_features_with_classification_gpu(
            expanded_pcl,
            curvature_threshold=self.curvature_threshold,
            gradient_threshold=self.gradient_threshold,
            voxel_size=self.key_voxel_size,
            target_num_points=self.inducing_points,
            k=self.knn_k,
            return_aux=True
        )

        self.keypoints = key_points
        robot_points = self.generate_robot_points(self.x_length, self.y_length)
        key_points = np.vstack((key_points, robot_points))
        # Robot footprint points have no source point either (c = g = 0), so pad with -1
        self.key_src_idx = np.concatenate((key_src_idx, np.full(len(robot_points), -1)))
        
        self.Xs = key_points[:, 0].reshape(-1, 1)
        self.Ys = key_points[:, 1].reshape(-1, 1)
        self.Zs = key_points[:, 2].reshape(-1, 1)
        self.curvatures = key_points[:, 4].reshape(-1, 1)
        self.gradients = key_points[:, 5].reshape(-1, 1)
        
        start_time = time.time()
        data = np.column_stack((self.Xs, self.Ys, self.curvatures, self.gradients))
        grid_train=None
        if self.open_pca:
            grid_train=self.pca.fit_transform(data)
        else:
            grid_train=data
        d_in = torch.tensor(grid_train, dtype=torch.float32, device=self.device)
        d_out = torch.tensor(self.Zs, dtype=torch.float32, device=self.device).squeeze()
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        if self.gp_noise_init > 0:
            # Start the (still learned) height noise here instead of GPyTorch's default 0.693 m^2
            likelihood.noise = self.gp_noise_init
        sgp_model = SGPModel(d_in, d_out, likelihood, self.inducing_points, self.lengthscale, self.alpha, self.apply_kernel_init).to(self.device)

        start_time = time.time()
        sgp_model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(sgp_model.parameters(), lr=self.gp_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, sgp_model)

        self.gp_loss_history = []
        for _ in range(self.gp_train_iters):
            optimizer.zero_grad()
            output = sgp_model(d_in)
            loss = -mll(output, d_out).mean()
            loss.backward()
            optimizer.step()
            self.gp_loss_history.append(loss.item())
        
        sgp_model.eval()
        likelihood.eval()
        
        self.sampling_grid()
        grid_test=None
        if self.open_pca:
            grid_test=self.pca.fit_transform(self.grid)
        else:
            grid_test=self.grid
        Xtest_tensor = torch.tensor(grid_test, dtype=torch.float32, requires_grad=True).to(sgp_model.device)
        start_time = time.time()
        preds = sgp_model.likelihood(sgp_model(Xtest_tensor))
        mean = preds.mean.detach().cpu().numpy()
        var = preds.variance.detach().cpu().numpy()
        grad_mean = torch.autograd.grad(preds.mean.sum(), Xtest_tensor, create_graph=True)[0].detach().cpu().numpy()
        
        filtered_mean, filtered_var, filtered_grad_mean, filter_grid = self.filter_low_uncertainty_data(mean, var, grad_mean, self.grid)
        
        start_time = time.time()
        self.mean = mean
        # Raw, full-length GP variance -- aligned 1:1 with self.mean/self.grid
        # (unlike filtered_var below, which drops cells and feeds only the
        # normalized "uncertainty" information-gain score used internally by
        # BGK fusion). Exposed for consumers that need the actual per-cell
        # variance alongside the published cost, e.g. closed-form Gaussian
        # risk metrics downstream.
        self.var = var
        self.grad_mean = filtered_grad_mean
        self.slope = self.analyzer.calculate_slope(filtered_grad_mean, self.max_slope, self.min_slope, self.test_slope)
        self.flatness = self.analyzer.calculate_flatness_entropy(np.array(filter_grid[:, 2]), self.max_flatness, self.min_flatness, self.test_flatness)
        self.step_height = self.analyzer.calculate_step_height_topology(np.array(filter_grid[:, 3]), self.max_height, self.min_height, self.test_height)
        self.uncertainty = self.analyzer.calculate_uncertainty_information_gain(filtered_var, 1, self.max_uncertainty, self.min_uncertainty, self.test_uncertainty)
        current_position = (pose[0], pose[1])
        
        start_time = time.time()
        self.traversability = 1.0 - self.analyzer.calculate_traversability(self.mean, self.slope, self.flatness, self.step_height, self.uncertainty, current_position, self.w_slope, self.w_flatness, self.w_step_height, self.bgk_threshold)

        if self.num_samples > 0:
            self.traversability_samples = self.sample_traversability_distribution(sgp_model)

    def sample_traversability_distribution(self, sgp_model):
        # N Monte Carlo samples of the cost per cell (high = bad, same orientation as
        # self.traversability). Each sample: LiDAR noise on the cloud, bootstrap of every point's
        # k-neighbourhood, re-interpolated k*/g*, and a spatial gradient drawn from the GP's
        # posterior derivative distribution. The GP itself is not retrained. Samples bypass BGK
        # temporal fusion (calculate_traversability mutates its history and min-max rescales).
        feature_rows = self.key_src_idx >= 0
        feature_src = torch.as_tensor(self.key_src_idx[feature_rows], device=self.pcl_tensor.device)
        samples = np.empty((len(self.keep_idx), self.num_samples), dtype=np.float32)

        for i in range(self.num_samples):
            noisy = choose_point.perturb_points_lidar(self.pcl_tensor, self.lidar_range_std, self.lidar_angular_std, self.sample_generator)
            curvatures_all, gradients_all = choose_point.bootstrap_features_gpu(noisy, self.knn_indices, self.sample_generator)

            curvatures_key = self.curvatures.copy()
            gradients_key = self.gradients.copy()
            curvatures_key[feature_rows, 0] = curvatures_all[feature_src].cpu().numpy()
            gradients_key[feature_rows, 0] = gradients_all[feature_src].cpu().numpy()
            curvatures_pred, gradients_pred = self._interp_features(curvatures_key, gradients_key)

            grid = np.column_stack((self.grid_xy, curvatures_pred, gradients_pred))
            X = torch.tensor(grid, dtype=torch.float32, device=sgp_model.device)
            mu_g, sigma_g = self._spatial_gradient_posterior(sgp_model, X)

            # Cholesky in float64: Sigma_g is only 2x2 per cell, and float32 can't resolve a
            # 1e-8 jitter against O(1e-2) entries
            eye = torch.eye(2, dtype=torch.float64, device=sigma_g.device)
            L, info = torch.linalg.cholesky_ex(sigma_g.double() + self.grad_cov_jitter * eye)
            if (info > 0).any():
                print(f"[sample_traversability_distribution] Cholesky failed for {int((info > 0).sum())} cells; their gradient noise is dropped")
                L[info > 0] = 0.0
            z = torch.randn(len(grid), 2, 1, generator=self.sample_generator, device=L.device, dtype=torch.float64)
            grad_sample = (mu_g.double() + (L @ z).squeeze(-1)).cpu().numpy()[self.keep_idx]

            if self.smooth_sample_slope:
                slope = self.analyzer.calculate_slope(grad_sample, self.max_slope, self.min_slope)
            else:
                slope = self.analyzer.normalize_attribute(np.linalg.norm(grad_sample, axis=1), self.min_slope, self.max_slope)
            flatness = self.analyzer.calculate_flatness_entropy(grid[self.keep_idx, 2], self.max_flatness, self.min_flatness)
            step_height = self.analyzer.calculate_step_height_topology(grid[self.keep_idx, 3], self.max_height, self.min_height)

            samples[:, i] = slope * self.w_slope + flatness * self.w_flatness + step_height * self.w_step_height

        return samples

    def _spatial_gradient_posterior(self, sgp_model, X):
        # Posterior of the spatial gradient (dz/dx, dz/dy) of the latent GP at each row of X:
        # mean mu_g (M, 2) and covariance Sigma_g (M, 2, 2), where
        # Sigma_g[i] = d^2 cov(a, b) / da_xy db_xy at a = b = X[i], from the posterior covariance.
        # A and B are separate copies so the mixed derivative can be taken; the diagonal of the
        # A-B cross block couples only A[i] with B[i], so summing it before each backward pass
        # gives every cell's own 2x2 block at once.
        M = X.shape[0]
        A = X.clone().requires_grad_(True)
        B = X.clone().requires_grad_(True)
        post = sgp_model(torch.cat([A, B]))

        mu_g = torch.autograd.grad(post.mean[:M].sum(), A, retain_graph=True)[0][:, :2]
        cross = post.covariance_matrix[:M, M:].diagonal()
        grad_a = torch.autograd.grad(cross.sum(), A, create_graph=True)[0]
        rows = [torch.autograd.grad(grad_a[:, d].sum(), B, retain_graph=(d == 0))[0][:, :2] for d in range(2)]
        sigma = torch.stack(rows, dim=1).detach()

        # Symmetrise and clamp tiny negative eigenvalues from float32 round-off
        sigma = 0.5 * (sigma + sigma.transpose(1, 2))
        evals, evecs = torch.linalg.eigh(sigma)
        sigma = evecs @ torch.diag_embed(evals.clamp_min(0)) @ evecs.transpose(1, 2)
        return mu_g.detach(), sigma

    def update_map(self, pose, local_pointcloud):
        local_pointcloud = np.array(local_pointcloud)
        self.generate_local_traversability_map(pose, local_pointcloud)