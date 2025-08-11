from setuptools import setup
import os

# Load requirements.txt
package_name = "drone_eye_detector"
# __file__ is in build/drone_eye_detector/setup.py during build
# We want to read requirements.txt from source folder: ../src/drone_eye_detector/requirements.txt

build_dir = os.path.abspath(os.path.dirname(__file__))          # e.g. /home/ros2/ws/build/drone_eye_detector
ws_dir = os.path.dirname(os.path.dirname(build_dir))             # /home/ros2/ws
req_path = os.path.join(ws_dir, "src", "drone_eye_detector", "requirements.txt")

with open(req_path) as f:
    requirements = f.read().splitlines()

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=requirements,
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="your_email@example.com",
    description="YOLO + Deep SORT detector for Drone Eye",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Example executable
            "detector_node = drone_eye_detector.detector_node:main"
        ],
    },
)
