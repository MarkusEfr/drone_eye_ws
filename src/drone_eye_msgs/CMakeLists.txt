cmake_minimum_required(VERSION 3.8)
project(drone_eye_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/BoundingBox.msg"
  "msg/BoundingBoxes.msg"
  "msg/TrackingInfo.msg"
  "msg/TrackingArray.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
)

ament_package()
