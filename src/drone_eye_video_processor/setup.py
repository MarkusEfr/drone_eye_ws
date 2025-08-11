from setuptools import find_packages, setup

package_name = "drone_eye_video_processor"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ros2",
    maintainer_email="ros2@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "video_processor = drone_eye_video_processor.video_processor:main",
            "socket_server = drone_eye_video_processor.socket_server:main",
        ],
    },
)
