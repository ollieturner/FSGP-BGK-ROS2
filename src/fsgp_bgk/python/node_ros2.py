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
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from tf2_ros import Buffer, TransformListener, TransformException
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import numpy as np
from f_sgp_bgk import TraversabilityAnalyzer  
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import yaml
# from scipy.interpolate import RegularGridInterpolator     # Not used while publish_resolution == resolution (see publish_global_grid)

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

        self.global_pose = None         # Pose of the last processed cloud (from TF at its stamp)
        self.latest_pcl = None  
        # Added: stamp of the last processed cloud, so each cloud is processed once (docs/pose_cloud_sync_plan.md)
        self.last_cloud_stamp = None

        # Added: pose for each cloud comes from TF map -> base_link at the cloud's stamp (same source as trial_data_logger)
        # - Sim publishes this TF at 50 Hz and builds each cloud from the latest TF pose, stamped at publish
        self.robot_frame = 'base_link'
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)     # Own ReentrantCallbackGroup
        # Wait for TF to reach the cloud's stamp: the TF buffer measured 0.4-1.5 s behind at frame start (its Python
        # callbacks compete for the GIL during map_thread's compute), so 0.1 s skipped about half the frames
        self.tf_timeout = Duration(seconds=2.0)

        # Added: callback groups for MultiThreadedExecutor (see main) so subscriptions keep running while map_thread computes
        # - Single-threaded spin starved them: callbacks ran at most once per ~4 s frame (see plan's step 0 results)
        self.map_group = MutuallyExclusiveCallbackGroup()   # map_thread never overlaps itself
        self.sub_group = MutuallyExclusiveCallbackGroup()   # cloud subscription

        # DIAG step 0 (docs/pose_cloud_sync_plan.md): track how fresh the pose/cloud are each frame - remove after verifying
        self.diag_cloud_cbs = 0         # elevation_cb calls since the previous frame
        self.diag_prev_pose = None      # Pose used by the previous frame

        self.grid_resolution = self.analyzer.resolution
        # Changed: original node grid was 24 x 24 (int(5 // 0.2), float floor) while the analyser's is 25 x 25, so cells
        # didn't line up. Now uses the analyser's own axes (cell centres -2.5 ... 2.3 m from the robot)
        # self.grid_size = (int(self.max_radius * 2 // self.grid_resolution), int(self.max_radius * 2 // self.grid_resolution))
        # self.grid_half = int(self.max_radius // self.grid_resolution)
        self.grid_x = self.analyzer.grid_x
        self.grid_y = self.analyzer.grid_y
        self.grid_size = (len(self.grid_x), len(self.grid_y))
        # Map-frame offset of every published cell from the robot (axis-aligned with the map, centred on the robot)
        self.cell_dx, self.cell_dy = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')

        # Publishing the grid directly needs both resolutions equal (no interpolation step anymore)
        if not np.isclose(self.publish_resolution, self.grid_resolution):
            raise ValueError(f"publish_resolution ({self.publish_resolution}) must equal resolution ({self.grid_resolution}); "
                             "see docs/grid_rotation_fix_plan.md step 5")

        # Changed: to store per-cell traversability samples
        # - No fusion and smoothing to preserve distribution (removed log_odds etc)
        # - Fully rewritten each frame (unseen cells = 1, lethal)
        self.global_sample_grid = np.ones((*self.grid_size, self.num_samples), dtype=np.float32)

        # Grid for per-factor distributions for each cell, if enabled
        self.publish_risk_factors = self.analyzer.publish_risk_factors
        if self.publish_risk_factors:
            self.global_factor_grid = np.full((*self.grid_size, 3 * self.num_samples), np.nan, dtype=np.float32)

        # Resolution settings as original
        # Commented out: publish_global_grid now publishes the grid cells directly (publish_resolution == resolution).
        # Kept for if publish_resolution is ever made finer than resolution again (would need the interpolation back)
        # self.high_res_resolution =  self.publish_resolution
        # self.high_res_half = int(self.max_radius / self.high_res_resolution)
        # self.high_res_shape = (self.high_res_half * 2, self.high_res_half * 2)
        # self.high_res_x = (np.arange(self.high_res_shape[0]) - self.high_res_half) * self.high_res_resolution
        # self.high_res_y = (np.arange(self.high_res_shape[1]) - self.high_res_half) * self.high_res_resolution
        # self.high_res_xx, self.high_res_yy = np.meshgrid(self.high_res_x, self.high_res_y, indexing='ij')
        # self.high_res_points = np.column_stack((self.high_res_xx.flatten(), self.high_res_yy.flatten()))

        # Define subscribers to point cloud and odometry topics
        self.sph_pcl_sub = self.create_subscription(PointCloud2, self.cloud_topic, self.elevation_cb, 3,
                                                    callback_group=self.sub_group)
        # Commented out: pose now from TF at each cloud's stamp; /odom backlog gave poses up to 16 s old (see plan)
        # self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 3)

        # Changed: Kept point cloud publisher and added one for risk factors
        self.traversability_pcl_ldd_pub = self.create_publisher(PointCloud2, "traversability_pcl_ldd", 3)
        if self.publish_risk_factors:
            self.risk_factor_pcl_pub = self.create_publisher(PointCloud2, "risk_factor_pcl", 3)

        self.map_timer = self.create_timer(self.map_update_rate, self.map_thread, callback_group=self.map_group)

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
    # Not used: /odom subscription commented out, pose comes from _lookup_pose (TF at the cloud's stamp)
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
            self.diag_cloud_cbs += 1        # DIAG step 0

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

    # Added: Place a robot-frame local grid into the map-aligned, robot-centred grid by inverse mapping (yaw only)
    # - For each map cell, rotate its offset from the robot into the robot frame and take the nearest local cell
    # - Every cell gets exactly one value each frame (no holes, duplicates or stale cells); outside the local square = fill
    # - local_values: (nx, ny, ...) indexed [ix, iy] on the analyser's grid_x/grid_y
    def _local_to_map_grid(self, local_values, yaw, fill):
        # Rotate map-frame offsets into the robot frame (transpose of the robot's yaw rotation)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        lx = cos_yaw * self.cell_dx + sin_yaw * self.cell_dy
        ly = -sin_yaw * self.cell_dx + cos_yaw * self.cell_dy

        # Nearest local cell
        ix = np.rint((lx - self.grid_x[0]) / self.grid_resolution).astype(int)
        iy = np.rint((ly - self.grid_y[0]) / self.grid_resolution).astype(int)
        valid = (ix >= 0) & (ix < len(self.grid_x)) & (iy >= 0) & (iy < len(self.grid_y))

        # Gather values, fill cells that rotate outside the local grid
        map_values = np.full(local_values.shape, fill, dtype=np.float32)
        map_values[valid] = local_values[ix[valid], iy[valid]]
        return map_values

    # Write frame's map from map-frame into the robot-centred grid in map frame
    # Not used: replaced by _local_to_map_grid (scattering with .astype(int) left holes/stale cells when rotated and
    # mismatched the 25 vs 24 grids). Kept for reference, see docs/grid_rotation_fix_plan.md
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
        # Changed: snapshot the latest cloud (elevation_cb can replace it mid-frame now it runs on another thread)
        cloud = self.latest_pcl
        if cloud is None:
            return
        # Process each cloud once (skip if no new cloud since the last frame)
        stamp = cloud.header.stamp
        if self.last_cloud_stamp is not None and (stamp.sec, stamp.nanosec) == self.last_cloud_stamp:
            return

        # Pose the cloud was captured at; skip this frame if TF doesn't cover the stamp yet
        diag_tf_start = self.get_clock().now()      # DIAG step 0
        pose, pose_stamp = self._lookup_pose(stamp)
        diag_tf_wait = (self.get_clock().now() - diag_tf_start).nanoseconds / 1e9     # DIAG step 0
        if pose is None:
            return
        self.last_cloud_stamp = (stamp.sec, stamp.nanosec)
        self.global_pose = pose

        # DIAG step 0 (docs/pose_cloud_sync_plan.md): cloud age, pose vs cloud stamp, callbacks since last frame
        diag_start = self.get_clock().now()
        diag_cloud_age = (diag_start - Time.from_msg(stamp)).nanoseconds / 1e9
        diag_pose_age = (diag_start - Time.from_msg(pose_stamp)).nanoseconds / 1e9
        diag_pose_vs_cloud = (Time.from_msg(pose_stamp) - Time.from_msg(stamp)).nanoseconds / 1e9
        diag_pose_moved = (float('nan') if self.diag_prev_pose is None else
                           float(np.hypot(pose[0] - self.diag_prev_pose[0], pose[1] - self.diag_prev_pose[1])))
        diag_msg = (f'[sync diag] cloud_age={diag_cloud_age:.3f}s pose_age={diag_pose_age:.3f}s '
                    f'pose_vs_cloud={diag_pose_vs_cloud:.3f}s tf_wait={diag_tf_wait:.3f}s cloud_cbs={self.diag_cloud_cbs} '
                    f'pose_moved={diag_pose_moved:.3f}m pose=({pose[0]:.3f}, {pose[1]:.3f})')
        self.diag_cloud_cbs = 0
        self.diag_prev_pose = list(pose)

        # Turn point cloud into (x, y, z) array
        local_points_np = self.pointcloud2_to_xyz(cloud)

        # Analyse point cloud with FSGP-BGK analyser (traversability costs etc)
        self.analyzer.update_map(pose, local_points_np)

        # Each cell's (x, y) position and predicted height (mean)
        # grid = self.analyzer.grid
        # mean = self.analyzer.mean
        keep = self.analyzer.keep_idx       # Align with length of samples
        # grid = self.analyzer.grid[keep]
        # mean = self.analyzer.mean[keep]

        # # Store x, y, GP elevation mean, then the N traversability samples per cell in row format
        # columns = [grid[:, 0], grid[:, 1], mean, self.analyzer.traversability_samples]
        # # Add risk factor samples per cell if enabled
        # if self.publish_risk_factors:
        #     columns += [self.analyzer.slope_samples, self.analyzer.flatness_samples, self.analyzer.step_height_samples]
        # smpld_pcl = np.column_stack(columns)

        # # Rotate and move x, y and height columns into the map frame (samples unchanged)
        # position = np.array(self.global_pose[:3])
        # orientation = np.array(quaternion_from_euler(*self.global_pose[3:]))
        # global_smpld_pcl = self.transform_smpl_pcl(smpld_pcl, position, orientation)

        # # Write samples into the map-aligned, robot-centred grid
        # self.update_global_grid(global_smpld_pcl)

        # Changed: replaces the transform + scatter above with inverse mapping (see docs/grid_rotation_fix_plan.md)
        # Rebuild full local grids (nx, ny, N) on the analyser's axes; cells dropped by the uncertainty filter are NaN
        nx, ny = self.grid_size
        local_samples = np.full((nx * ny, self.num_samples), np.nan, dtype=np.float32)
        local_samples[keep] = self.analyzer.traversability_samples
        local_samples = local_samples.reshape(nx, ny, self.num_samples)

        # Rotate into the map-aligned grid by yaw only (2.5D map); outside the local square and filtered cells = 1 (lethal, as original)
        yaw = pose[5]
        self.global_sample_grid = self._local_to_map_grid(local_samples, yaw, fill=1.0)
        self.global_sample_grid[np.isnan(self.global_sample_grid)] = 1.0

        # Same for risk factor samples if enabled (unknown stays NaN so factors don't recombine incorrectly)
        if self.publish_risk_factors:
            local_factors = np.full((nx * ny, 3 * self.num_samples), np.nan, dtype=np.float32)
            local_factors[keep] = np.column_stack([self.analyzer.slope_samples, self.analyzer.flatness_samples,
                                                   self.analyzer.step_height_samples])
            local_factors = local_factors.reshape(nx, ny, 3 * self.num_samples)
            self.global_factor_grid = self._local_to_map_grid(local_factors, yaw, fill=np.nan)

        # Publish the traversability and risk factor sample grids (at the cloud's pose, stamped with its capture time)
        self.publish_global_grid(pose, stamp)

        # DIAG step 0: compute time = how stale the published map is when it goes out (stamp is now the cloud's time)
        diag_compute = (self.get_clock().now() - diag_start).nanoseconds / 1e9
        self.get_logger().info(f'{diag_msg} compute={diag_compute:.3f}s')


    # Edited: Publish the sample grid as a point cloud in map frame (one point per grid cell)
    # - Original computed on resolution (0.2m) and interpolated to publish_resolution (0.1m). Now both 0.2m, so cells
    #   are published directly; the interpolation is commented out below (with the grid now on the analyser's axes
    #   -2.5 ... 2.3, the old publish points -2.4 ... 2.2 would sit halfway between cells and blend samples)
    # Changed: pose and stamp passed in (pose = TF at the cloud's stamp) rather than read from self.global_pose
    def publish_global_grid(self, pose, stamp):

        # # Extract the map's shape and the positions of the grid's cells
        # grid_shape = self.global_sample_grid.shape
        # low_res_x = (np.arange(grid_shape[0]) - self.grid_half) * self.grid_resolution + self.global_pose[0]
        # low_res_y = (np.arange(grid_shape[1]) - self.grid_half) * self.grid_resolution + self.global_pose[1]

        # # Interpolate to a finer grid
        # try:
        #     # Extract current fine points
        #     high_res_points_global = self.high_res_points + self.global_pose[:2]

        #     # Interpolate estimate values at every find point (outside get cost 1)
        #     interp_samples = RegularGridInterpolator((low_res_x, low_res_y), self.global_sample_grid, method='linear', bounds_error=False, fill_value=1)
        #     # interp_samples = RegularGridInterpolator((low_res_x, low_res_y), self.global_sample_grid, method='nearest', bounds_error=False, fill_value=1)      # Changed to nearest for better retention of independence
        #     high_res_samples = interp_samples(high_res_points_global)

        #     # Repeated for risk factor distributions
        #     if self.publish_risk_factors:
        #         # Unknown is NaN rather than 1 so the factors don't recombine incorrectly
        #         interp_factors = RegularGridInterpolator((low_res_x, low_res_y), self.global_factor_grid, method='linear', bounds_error=False, fill_value=np.nan)
        #         # interp_factors = RegularGridInterpolator((low_res_x, low_res_y), self.global_factor_grid, method='nearest', bounds_error=False, fill_value=np.nan)        # Changed to nearest for better retention of independence
        #         high_res_factors = interp_factors(high_res_points_global)
        # except Exception as e:
        #     self.get_logger().error(f"Interpolation failed: {e}")
        #     return

        # Added: grid cells published directly, offsets flattened in the same [ix, iy] order as the grids
        cell_dx = self.cell_dx.flatten()
        cell_dy = self.cell_dy.flatten()
        high_res_samples = self.global_sample_grid.reshape(-1, self.num_samples).copy()
        if self.publish_risk_factors:
            high_res_factors = self.global_factor_grid.reshape(-1, 3 * self.num_samples).copy()

        # Assign flat sheet height to every point - isn't used downstream
        # - Makes the costmap move at ground level below the sensor
        # - TODO Could change to make costmap follow the terrain in RViz by using mean
        z = np.full_like(cell_dx, pose[2]) + self.analyzer.base_height

        # Points at the boundary are set to cost of 1
        radius = np.hypot(cell_dx, cell_dy)
        edge = radius > self.max_radius - self.publish_resolution * 2
        high_res_samples[edge] = 1

        # Build point cloud data
        high_res_cloud_data = np.column_stack([cell_dx + pose[0],
                                               cell_dy + pose[1],
                                               z, high_res_samples])
        
        # Assign a stamp (use same for risk factors so they are paired exactly on receiving end)
        # Changed: cloud's capture time instead of now(), so downstream TF lookups (logger) get the pose the map was built at
        # self.header.stamp = self.get_clock().now().to_msg()
        self.header.stamp = stamp
        self.traversability_pcl_ldd_pub.publish(pc2.create_cloud(self.header, self.fields, high_res_cloud_data))

        # Publish risk factor data if enabled
        if self.publish_risk_factors:
            high_res_factors[edge] = np.nan
            high_res_factor_data = np.column_stack([high_res_cloud_data[:, :3], high_res_factors])
            self.risk_factor_pcl_pub.publish(pc2.create_cloud(self.header, self.factor_fields, high_res_factor_data))

    # Added: robot pose [x, y, z, roll, pitch, yaw] in the map frame at a stamp, from TF (same format as odom_cb)
    # - Returns (pose, transform stamp), or (None, None) if TF doesn't cover the stamp yet (frame is skipped)
    def _lookup_pose(self, stamp_msg):
        try:
            tf = self.tf_buffer.lookup_transform(self.global_link, self.robot_frame, Time.from_msg(stamp_msg),
                                                 timeout=self.tf_timeout)
        except TransformException as e:
            self.get_logger().warn(f'No TF {self.global_link} -> {self.robot_frame} at cloud stamp, skipping frame: {e}',
                                   throttle_duration_sec=5.0)
            return None, None
        t = tf.transform.translation
        q = tf.transform.rotation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return [t.x, t.y, t.z, roll, pitch, yaw], tf.header.stamp

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
    # Changed: multi-threaded so the cloud and TF subscriptions run during map_thread (was rclpy.spin(node), single thread)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()