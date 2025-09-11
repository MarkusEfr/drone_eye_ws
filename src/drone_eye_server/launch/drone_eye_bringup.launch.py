from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="drone_eye_server",
                executable="server_node",
                name="server",
                output="screen",
            ),
            Node(
                package="drone_eye_detector",
                executable="detector_node",
                name="detector",
                output="screen",
            ),
            Node(
                package="drone_eye_tracker",
                executable="tracker_node",
                name="tracker",
                output="screen",
            ),
            Node(
                package="drone_eye_visualizer",
                executable="visualizer_node",
                name="visualizer",
                output="screen",
            ),
            Node(
                package="drone_eye_navigation",
                executable="orbslam3_node",
                name="orbslam3",
                output="screen",
                parameters=[
                    {"vocab_file": "/home/ros2/ws/src/ORB_SLAM3/Vocabulary/ORBvoc.txt"},
                    {
                        "config_file": "/home/ros2/ws/src/ORB_SLAM3/configs/your_camera.yaml"
                    },
                ],
            ),
        ]
    )
