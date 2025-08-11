ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base-jammy

########################
# System Setup (root)
########################

# Create non-root user with sudo access
RUN useradd -m ros2 && \
    echo "ros2:ros2" | chpasswd && \
    adduser ros2 sudo && \
    echo "ros2 ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install base tools and dependencies
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git curl lsb-release \
    libgl1 libglib2.0-0 sudo && \
    pip install --upgrade pip argcomplete

# Install Python dependencies
COPY ./src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download YOLOv8m model and fix permissions
RUN mkdir -p /home/ros2/ws/models && \
    curl -L -o /home/ros2/ws/models/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt && \
    chown -R ros2:ros2 /home/ros2/ws

########################
# User Setup
########################

# Switch to non-root user and bash shell
SHELL ["/bin/bash", "-c"]
USER ros2
WORKDIR /home/ros2/ws

# Copy ROS 2 source code into workspace
COPY --chown=ros2:ros2 ./src ./src

# Source ROS and build the workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build

########################
# Entry (Optional)
########################

# Automatically source workspace on login (optional quality-of-life)
RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> ~/.bashrc && \
    echo 'source ~/ws/install/setup.bash' >> ~/.bashrc
