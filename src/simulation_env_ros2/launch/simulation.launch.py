# ROS 2 launch file that brings up simulation environment visualisation 
# Simulation backend, fake LiDAR publisher and pre-configured RViz visualiser

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the shared directory path of the package.
    pkg_share_dir = get_package_share_directory('simulation_env')

    # Define the path to the RViz configuration file.
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'simulation.rviz')

    # Define the path to the parameter file.
    params_file = os.path.join(pkg_share_dir, 'config', 'params.yaml')

    # Define nodes
    simulation_node = Node(
        package='simulation_env',
        executable='simulation_node',
        name='simulation_node',
        output='screen',
        parameters=[params_file]  # Load parameter file
    )

    simulated_lidar_node = Node(
        package='simulation_env',
        executable='simulated_lidar',
        name='simulated_lidar',
        output='screen',
        # parameters=[params_file]  # Load parameter file
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # Return LaunchDescription
    return LaunchDescription([
        simulation_node,
        simulated_lidar_node,
        rviz_node
    ])