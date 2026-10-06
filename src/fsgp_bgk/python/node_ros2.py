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

# Modified FSGP-BGK node to publish per-cell traversability distributions (and risk factor distributions when enabled)
# - LiDAR point cloud is run through TraversabilityAnalyser (GP terrain and Monte Carlo sampling) to produce distributions
# - Distributions placed onto grid centred on robot then published

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
        # Get config parameters
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

        # Start Traversability Analyser object
        self.analyzer = TraversabilityAnalyzer(config_path=config_path)
        self.max_radius = (self.analyzer.x_length + self.analyzer.y_length) / 4

        # Added: extended for traversability distributions
        self.num_samples = self.analyzer.num_samples
        if self.num_samples < 1:
            # The node only publishes the per-cell traversability distribution
            raise ValueError("num_samples must be >= 1")

        self.global_pose = None  
        self.latest_pcl = None  

        self.grid_resolution = self.analyzer.resolution  
        self.grid_size = (int(self.max_radius * 2 // self.grid_resolution), int(self.max_radius * 2 // self.grid_resolution))
        self.grid_half = int(self.max_radius // self.grid_resolution)
        
        # Changed: to store per-cell traversability samples
        # - No fusion and smoothing to preserve distribution (removed log_odds etc)
        self.global_sample_grid = np.zeros((*self.grid_size, self.num_samples), dtype=np.float32)
        
        # Grid for per-factor distributions for each cell, if enabled
        self.publish_risk_factors = self.analyzer.publish_risk_factors
        if self.publish_risk_factors:
            self.global_factor_grid = np.zeros((*self.grid_size, 3 * self.num_samples), dtype=np.float32)

        # Resolution settings as original
        self.high_res_resolution =  self.publish_resolution
        self.high_res_half = int(self.max_radius / self.high_res_resolution)
        self.high_res_shape = (self.high_res_half * 2, self.high_res_half * 2)
        self.high_res_x = (np.arange(self.high_res_shape[0]) - self.high_res_half) * self.high_res_resolution
        self.high_res_y = (np.arange(self.high_res_shape[1]) - self.high_res_half) * self.high_res_resolution
        self.high_res_xx, self.high_res_yy = np.meshgrid(self.high_res_x, self.high_res_y, indexing='ij')
        self.high_res_points = np.column_stack((self.high_res_xx.flatten(), self.high_res_yy.flatten()))

        # Define subscribers to point cloud and odometry topics
        self.sph_pcl_sub = self.create_subscription(PointCloud2, self.cloud_topic, self.elevation_cb, 3)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 3)

        # Changed: Kept point cloud publisher and added one for risk factors
        self.traversability_pcl_ldd_pub = self.create_publisher(PointCloud2, "traversability_pcl_ldd", 3)
        if self.publish_risk_factors:
            self.risk_factor_pcl_pub = self.create_publisher(PointCloud2, "risk_factor_pcl", 3)

        self.map_timer = self.create_timer(self.map_update_rate, self.map_thread)

        self.header = Header()
        self.header.frame_id = self.global_link
        # Changed: x, y, z, then one float32 per traversability sample (cost, high = bad)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ] + [
            PointField(name=f'trav_{i}', offset=12 + 4 * i, datatype=PointField.FLOAT32, count=1)
            for i in range(self.num_samples)
        ]
        # Added: similar for risk factor publishing - risk_factor_pcl: x, y, z, then slope_i, flat_i, step_i (normalised [0, 1] factor samples, NaN = no factor data)
        if self.publish_risk_factors:
            self.factor_fields = self.fields[:3] + [
                PointField(name=f'{name}_{i}', offset=12 + 4 * (block * self.num_samples + i),
                           datatype=PointField.FLOAT32, count=1)
                for block, name in enumerate(('slope', 'flat', 'step'))
                for i in range(self.num_samples)
            ]

    # Extract latest pose from odometry
    def odom_cb(self, msg):
        # Skip if message is empty
        if msg is None:
            return

        # Read in position and orientation
        position = msg.pose.pose.position  
        orientation = msg.pose.pose.orientation  

        try:
            # Convert quaternion to roll, pitch and yaw for pose
            roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.global_pose = [position.x, position.y, position.z, roll, pitch, yaw]
        except Exception as ex:
            self.get_logger().warn(f'err: {ex}')

    # Store the latest point cloud
    def elevation_cb(self, msg):
        if msg is not None:
            self.latest_pcl = msg

    # Convert point cloud in PointCloud2 format to (x, y, z) array
    def pointcloud2_to_xyz(self, msg):
        # Extract points
        points = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

        # Initialise storage
        xyz = np.zeros((points.shape[0], 3), dtype=np.float32)

        # Fill in x, y, z data - each field is a 4-byte float32 starting at its offset in the row
        xyz[:, 0] = points[:, msg.fields[0].offset:msg.fields[0].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 1] = points[:, msg.fields[1].offset:msg.fields[1].offset + 4].view(np.float32).reshape(-1)
        xyz[:, 2] = points[:, msg.fields[2].offset:msg.fields[2].offset + 4].view(np.float32).reshape(-1)

        # Keep only points below the max_cloud_height and within the max_radius of the robot (FSGP-BGK's local costmap window)
        radius_sq = xyz[:, 0]**2 + xyz[:, 1]**2
        mask = (xyz[:, 2] < self.max_height) & (radius_sq < self.max_radius**2)
        return xyz[mask]

    # Note: removed observation_model since not fusing distributions

    # Write frame's map from map-frame into the robot-centred grid in map frame
    # TODO Original doesn't account for rotation? See plan for suggested improvements
    def update_global_grid(self, global_smpld_pcl):
        # Extract robot pose
        pose_x, pose_y = self.global_pose[0], self.global_pose[1]

        # Find each point's offset (in cells) from the robot (by subtracting pose)
        # - Move each point to coordinates for if robot was (0, 0)
        grid_indices = ((global_smpld_pcl[:, :2] - np.array([pose_x, pose_y])) / self.grid_resolution).astype(int)
        # TODO Fix 24 v 25 issue by implementing the fixed grid_rotation_plan.md thing
        
        # Keep points inside the grid when robot is moved to the centre of the array
        valid_indices = (grid_indices[:, 0] + self.grid_half >= 0) & (grid_indices[:, 0] + self.grid_half < self.grid_size[0]) & \
                        (grid_indices[:, 1] + self.grid_half >= 0) & (grid_indices[:, 1] + self.grid_half < self.grid_size[1])
        grid_indices = grid_indices[valid_indices]

        # Removed Bayesian fusion and smoothing for distribution and to not mix (retaining independence)

        ## Changed from original (to account for distribution):
        # Take samples for each point
        samples = global_smpld_pcl[valid_indices, 3:3 + self.num_samples]

        # Move the robot into the middle of the array
        x_indices = grid_indices[:, 0] + self.grid_half
        y_indices = grid_indices[:, 1] + self.grid_half
        # Write each cell's N samples with new frame's data
        self.global_sample_grid[x_indices, y_indices, :] = samples
        # Write risk factor samples in same way if enabled
        if self.publish_risk_factors:
            self.global_factor_grid[x_indices, y_indices, :] = global_smpld_pcl[valid_indices, 3 + self.num_samples:]

    # Edited from original: Process the latest cloud
    # - Run FSGP-BGK to get each local cell's cost samples, move them into the map frame, store them in the robot-centred grid and publish it
    def map_thread(self):
        # Skip if point cloud or pose don't exist
        if self.latest_pcl is None or self.global_pose is None:
            return

        # Turn point cloud into (x, y, z) array
        local_points_np = self.pointcloud2_to_xyz(self.latest_pcl)

        # Analyse point cloud with FSGP-BGK analyser (traversability costs etc)
        self.analyzer.update_map(self.global_pose, local_points_np)

        # Each cell's (x, y) position and predicted height (mean)
        # grid = self.analyzer.grid
        # mean = self.analyzer.mean
        keep = self.analyzer.keep_idx       # Align with length of samples
        grid = self.analyzer.grid[keep]
        mean = self.analyzer.mean[keep]

        # Store x, y, GP elevation mean, then the N traversability samples per cell in row format
        columns = [grid[:, 0], grid[:, 1], mean, self.analyzer.traversability_samples]
        # Add risk factor samples per cell if enabled
        if self.publish_risk_factors:
            columns += [self.analyzer.slope_samples, self.analyzer.flatness_samples, self.analyzer.step_height_samples]
        smpld_pcl = np.column_stack(columns)

        # Rotate and move x, y and height columns into the map frame (samples unchanged)
        position = np.array(self.global_pose[:3])
        orientation = np.array(quaternion_from_euler(*self.global_pose[3:]))
        global_smpld_pcl = self.transform_smpl_pcl(smpld_pcl, position, orientation)

        # Write samples into the map-aligned, robot-centred grid
        self.update_global_grid(global_smpld_pcl)

        # Publish the traversability and risk factor sample grids
        self.publish_global_grid()


    # Edited: Interpolate sample grid into finer point cloud in map frame then publish 
    # TODO check when updating resolution
    # - FSGP-BGK worked on 0.2m scale, but publishes here on a 0.1m scale to look smoother for planners and RViz?
    def publish_global_grid(self):
        # Skip if no pose available
        if self.global_pose is None:
            self.get_logger().warn("Global pose is not available.")
            return

        # Extract the map's shape and the positions of the grid's cells
        grid_shape = self.global_sample_grid.shape
        low_res_x = (np.arange(grid_shape[0]) - self.grid_half) * self.grid_resolution + self.global_pose[0]
        low_res_y = (np.arange(grid_shape[1]) - self.grid_half) * self.grid_resolution + self.global_pose[1]

        # Interpolate to a finer grid
        try:
            # Extract current fine points
            high_res_points_global = self.high_res_points + self.global_pose[:2]

            # Interpolate estimate values at every find point (outside get cost 1)
            # interp_samples = RegularGridInterpolator((low_res_x, low_res_y), self.global_sample_grid, method='linear', bounds_error=False, fill_value=1)
            interp_samples = RegularGridInterpolator((low_res_x, low_res_y), self.global_sample_grid, method='nearest', bounds_error=False, fill_value=1)      # Changed to nearest for better retention of independence   
            high_res_samples = interp_samples(high_res_points_global)

            # Repeated for risk factor distributions
            if self.publish_risk_factors:
                # Unknown is NaN rather than 1 so the factors don't recombine incorrectly
                # interp_factors = RegularGridInterpolator((low_res_x, low_res_y), self.global_factor_grid, method='linear', bounds_error=False, fill_value=np.nan)
                interp_factors = RegularGridInterpolator((low_res_x, low_res_y), self.global_factor_grid, method='nearest', bounds_error=False, fill_value=np.nan)        # Changed to nearest for better retention of independence        
                high_res_factors = interp_factors(high_res_points_global)
        except Exception as e:
            self.get_logger().error(f"Interpolation failed: {e}")
            return

        # Assign flat sheet height to every point - isn't used downstream
        # - Makes the costmap move at ground level below the sensor
        # - TODO Could change to make costmap follow the terrain in RViz by using mean
        z = np.full_like(self.high_res_xx.flatten(), self.global_pose[2]) + self.analyzer.base_height

        # Points at the boundary are set to cost of 1
        radius = np.hypot(self.high_res_xx.flatten(), self.high_res_yy.flatten())
        edge = radius > self.max_radius - self.high_res_resolution * 2
        high_res_samples[edge] = 1

        # Build high resolution point cloud data
        high_res_cloud_data = np.column_stack([self.high_res_xx.flatten() + self.global_pose[0],
                                               self.high_res_yy.flatten() + self.global_pose[1],
                                               z, high_res_samples])
        
        # Assign a stamp (use same for risk factors so they are paired exactly on receiving end)
        self.header.stamp = self.get_clock().now().to_msg()
        self.traversability_pcl_ldd_pub.publish(pc2.create_cloud(self.header, self.fields, high_res_cloud_data))

        # Publish risk factor data if enabled
        if self.publish_risk_factors:
            high_res_factors[edge] = np.nan
            high_res_factor_data = np.column_stack([high_res_cloud_data[:, :3], high_res_factors])
            self.risk_factor_pcl_pub.publish(pc2.create_cloud(self.header, self.factor_fields, high_res_factor_data))

    # Original: Move each local cell from the robot's frame to the map frame
    def transform_smpl_pcl(self, smpl_pcl, position, orientation):
        # Take robot-centric points (x, y, z)
        points = smpl_pcl[:, :3]

        # Compute rotation from orientation quaternion
        rotation_matrix = quaternion_matrix(orientation)[:3, :3]

        # Rotate every point by the robot's orientation then shift by robot's position to move cells into map frame
        transformed_points = np.dot(points, rotation_matrix.T) + position

        # Put back together with unchanged sample values (edited from original to take all trailing columns)
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