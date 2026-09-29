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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import numpy as np
from f_sgp_bgk import TraversabilityAnalyzer  
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import yaml
from scipy.interpolate import RegularGridInterpolator

class FSGP_BGK_Node(Node):
    def __init__(self):
        super().__init__('FSGP_BGK_Node')
        config_path = "../config/params.yaml"
        
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except Exception as e:
            self.get_logger().error(f"Failed to load config file: {e}")
            return

        self.cloud_topic = config["cloud_topic"]  
        self.odom_topic = config["odom_topic"]   
        self.global_link = config["global_link"]   
        self.max_height = config["max_cloud_height"]
        self.map_update_rate = config["map_update_rate"]
        self.publish_resolution = config["publish_resolution"]

        self.analyzer = TraversabilityAnalyzer(config_path=config_path)
        self.num_samples = self.analyzer.num_samples
        if self.num_samples < 1:
            # The node only publishes the per-cell traversability distribution
            raise ValueError("num_samples must be >= 1")
        self.max_radius = (self.analyzer.x_length + self.analyzer.y_length) / 4

        self.global_pose = None  
        self.latest_pcl = None  

        self.grid_resolution = self.analyzer.resolution  
        self.grid_size = (int(self.max_radius * 2 // self.grid_resolution), int(self.max_radius * 2 // self.grid_resolution))
        self.grid_half = int(self.max_radius // self.grid_resolution)
        # Per-cell traversability samples, one channel per sample (dense rather than N sparse
        # matrices). Cells are overwritten each frame; no log-odds fusion (binarising each sample
        # would destroy the distribution) and no spatial smoothing (it would average independent
        # draws across neighbouring cells and shrink each cell's spread).
        self.global_sample_grid = np.zeros((*self.grid_size, self.num_samples), dtype=np.float32)
        # Optional: per-cell samples of the three normalised risk factors the cost is built from,
        # channels slope 0..N-1, flatness 0..N-1, step height 0..N-1; same treatment as
        # global_sample_grid (overwritten each frame, no smoothing)
        self.publish_risk_factors = self.analyzer.publish_risk_factors
        if self.publish_risk_factors:
            self.global_factor_grid = np.zeros((*self.grid_size, 3 * self.num_samples), dtype=np.float32)

        self.high_res_resolution =  self.publish_resolution
        self.high_res_half = int(self.max_radius / self.high_res_resolution)
        self.high_res_shape = (self.high_res_half * 2, self.high_res_half * 2)
        self.high_res_x = (np.arange(self.high_res_shape[0]) - self.high_res_half) * self.high_res_resolution
        self.high_res_y = (np.arange(self.high_res_shape[1]) - self.high_res_half) * self.high_res_resolution
        self.high_res_xx, self.high_res_yy = np.meshgrid(self.high_res_x, self.high_res_y, indexing='ij')
        self.high_res_points = np.column_stack((self.high_res_xx.flatten(), self.high_res_yy.flatten()))

        self.sph_pcl_sub = self.create_subscription(PointCloud2, self.cloud_topic, self.elevation_cb, 3)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 3)

        self.traversability_pcl_ldd_pub = self.create_publisher(PointCloud2, "traversability_pcl_ldd", 3)
        if self.publish_risk_factors:
            self.risk_factor_pcl_pub = self.create_publisher(PointCloud2, "risk_factor_pcl", 3)

        self.map_timer = self.create_timer(self.map_update_rate, self.map_thread)

        self.header = Header()
        self.header.frame_id = self.global_link
        # x, y, z, then one float32 per traversability sample (cost, high = bad)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ] + [
            PointField(name=f'trav_{i}', offset=12 + 4 * i, datatype=PointField.FLOAT32, count=1)
            for i in range(self.num_samples)
        ]
        # risk_factor_pcl: x, y, z, then slope_i, flat_i, step_i (normalised [0, 1] factor samples,
        # NaN = no factor data), each block of N float32s
        if self.publish_risk_factors:
            self.factor_fields = self.fields[:3] + [
                PointField(name=f'{name}_{i}', offset=12 + 4 * (block * self.num_samples + i),
                           datatype=PointField.FLOAT32, count=1)
                for block, name in enumerate(('slope', 'flat', 'step'))
                for i in range(self.num_samples)
            ]

    def odom_cb(self, msg):
        if msg is None:
            return

        position = msg.pose.pose.position  
        orientation = msg.pose.pose.orientation  

        try:
            roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.global_pose = [position.x, position.y, position.z, roll, pitch, yaw]
        except Exception as ex:
            self.get_logger().warn(f'err: {ex}')

    def elevation_cb(self, msg):
        if msg is not None:
            self.latest_pcl = msg

    def pointcloud2_to_xyz(self, msg):
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        xyz = np.zeros((points.shape[0], 3), dtype=np.float32)
        xyz[:, 0] = points[:, msg.fields[0].offset:msg.fields[0].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 1] = points[:, msg.fields[1].offset:msg.fields[1].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 2] = points[:, msg.fields[2].offset:msg.fields[2].offset + 4].view(np.float32).reshape(-1)

        radius_sq = xyz[:, 0]**2 + xyz[:, 1]**2
        mask = (xyz[:, 2] < self.max_height) & (radius_sq < self.max_radius**2)
        return xyz[mask]

    def update_global_grid(self, global_smpld_pcl):
        pose_x, pose_y = self.global_pose[0], self.global_pose[1]
        grid_indices = ((global_smpld_pcl[:, :2] - np.array([pose_x, pose_y])) / self.grid_resolution).astype(int)
        valid_indices = (grid_indices[:, 0] + self.grid_half >= 0) & (grid_indices[:, 0] + self.grid_half < self.grid_size[0]) & \
                        (grid_indices[:, 1] + self.grid_half >= 0) & (grid_indices[:, 1] + self.grid_half < self.grid_size[1])
        grid_indices = grid_indices[valid_indices]
        samples = global_smpld_pcl[valid_indices, 3:3 + self.num_samples]

        x_indices = grid_indices[:, 0] + self.grid_half
        y_indices = grid_indices[:, 1] + self.grid_half
        self.global_sample_grid[x_indices, y_indices, :] = samples
        if self.publish_risk_factors:
            self.global_factor_grid[x_indices, y_indices, :] = global_smpld_pcl[valid_indices, 3 + self.num_samples:]

    def map_thread(self):
        if self.latest_pcl is None or self.global_pose is None:
            return

        local_points_np = self.pointcloud2_to_xyz(self.latest_pcl)

        self.analyzer.update_map(self.global_pose, local_points_np)
        grid = self.analyzer.grid
        mean = self.analyzer.mean

        # x, y, GP elevation mean, then the N traversability samples per cell
        columns = [grid[:, 0], grid[:, 1], mean, self.analyzer.traversability_samples]
        if self.publish_risk_factors:
            # ... then N slope, N flatness and N step-height samples per cell
            columns += [self.analyzer.slope_samples, self.analyzer.flatness_samples, self.analyzer.step_height_samples]
        smpld_pcl = np.column_stack(columns)

        position = np.array(self.global_pose[:3])
        orientation = np.array(quaternion_from_euler(*self.global_pose[3:]))
        global_smpld_pcl = self.transform_smpl_pcl(smpld_pcl, position, orientation)
        self.update_global_grid(global_smpld_pcl)

        self.publish_global_grid()

    def publish_global_grid(self):
        if self.global_pose is None:
            self.get_logger().warn("Global pose is not available.")
            return

        grid_shape = self.global_sample_grid.shape
        low_res_x = (np.arange(grid_shape[0]) - self.grid_half) * self.grid_resolution + self.global_pose[0]
        low_res_y = (np.arange(grid_shape[1]) - self.grid_half) * self.grid_resolution + self.global_pose[1]

        try:
            high_res_points_global = self.high_res_points + self.global_pose[:2]
            # Vector-valued grid (nx, ny, N) -> (P, N); unknown = lethal (cost 1)
            interp_samples = RegularGridInterpolator((low_res_x, low_res_y), self.global_sample_grid, method='linear', bounds_error=False, fill_value=1)
            high_res_samples = interp_samples(high_res_points_global)
            if self.publish_risk_factors:
                # Same linear interpolation (so interpolated trav still equals the weighted sum
                # of the interpolated factors); unknown = NaN rather than 1, since factors of 1
                # would recombine to w_slope + w_flatness + w_step_height, not a lethal cost
                interp_factors = RegularGridInterpolator((low_res_x, low_res_y), self.global_factor_grid, method='linear', bounds_error=False, fill_value=np.nan)
                high_res_factors = interp_factors(high_res_points_global)
        except Exception as e:
            self.get_logger().error(f"Interpolation failed: {e}")
            return

        z = np.full_like(self.high_res_xx.flatten(), self.global_pose[2]) + self.analyzer.base_height

        radius = np.hypot(self.high_res_xx.flatten(), self.high_res_yy.flatten())
        edge = radius > self.max_radius - self.high_res_resolution * 2
        high_res_samples[edge] = 1
        high_res_cloud_data = np.column_stack([self.high_res_xx.flatten() + self.global_pose[0],
                                               self.high_res_yy.flatten() + self.global_pose[1],
                                               z, high_res_samples])
        # One stamp for both clouds, so consumers can pair them exactly (TimeSynchronizer)
        self.header.stamp = self.get_clock().now().to_msg()
        self.traversability_pcl_ldd_pub.publish(pc2.create_cloud(self.header, self.fields, high_res_cloud_data))

        if self.publish_risk_factors:
            high_res_factors[edge] = np.nan
            high_res_factor_data = np.column_stack([high_res_cloud_data[:, :3], high_res_factors])
            self.risk_factor_pcl_pub.publish(pc2.create_cloud(self.header, self.factor_fields, high_res_factor_data))

    def transform_smpl_pcl(self, smpl_pcl, position, orientation):
        points = smpl_pcl[:, :3]
        rotation_matrix = quaternion_matrix(orientation)[:3, :3]
        transformed_points = np.dot(points, rotation_matrix.T) + position
        # [:, 3:] so every trailing column (the N traversability samples, and the factor
        # samples when publish_risk_factors is set) is carried through.
        transformed_smpl_pcl = np.column_stack((transformed_points, smpl_pcl[:, 3:]))
        return transformed_smpl_pcl

def main(args=None):
    rclpy.init(args=args)
    node = FSGP_BGK_Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()