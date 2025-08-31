from setuptools import setup, find_packages
import pathlib

package_name = "drone_eye_detector"

here = pathlib.Path(__file__).parent.resolve()
req_path = here / "requirements.txt"

if req_path.exists():
    with open(req_path) as f:
        install_requires = f.read().splitlines()
else:
    install_requires = []

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=install_requires,
    zip_safe=True,
    maintainer="ros2",
    maintainer_email="ros2@example.com",
    description="Drone Eye detector node (YOLOv8-based object detection)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = drone_eye_detector.detector_node:main",
        ],
    },
)
