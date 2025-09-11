cmake_minimum_required(VERSION 3.8)
project(drone_eye_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


add_compile_definitions(EIGEN_NO_DEPRECATED_WARNINGS)

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)


# === Dependencies ===
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)

# === ORB-SLAM3 paths ===
set(ORB_SLAM3_DIR ${CMAKE_SOURCE_DIR}/../ORB_SLAM3)

include_directories(
  include
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
  ${ORB_SLAM3_DIR}/Thirdparty/g2o
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${OPENGL_LIBRARIES}
)

link_directories(${ORB_SLAM3_DIR}/lib)

# === Executable ===
add_executable(orbslam3_node src/orbslam3_node.cpp)

ament_target_dependencies(orbslam3_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  OpenCV
)

target_link_libraries(orbslam3_node
  ORB_SLAM3
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${OPENGL_LIBRARIES}
)

# === Install ===
install(TARGETS
  orbslam3_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
