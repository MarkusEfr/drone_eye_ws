#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "cv_bridge/cv_bridge.h"
#include "System.h"

class ORBSLAM3Node : public rclcpp::Node
{
public:
    ORBSLAM3Node() : Node("orbslam3_node")
    {
        // Paths to ORB-SLAM3 vocabulary and config
        std::string vocab_file = this->declare_parameter<std::string>(
            "vocab_file", "/home/ros2/ws/src/ORB_SLAM3/Vocabulary/ORBvoc.txt");
        std::string config_file = this->declare_parameter<std::string>(
            "config_file", "/home/ros2/ws/src/ORB_SLAM3/configs/camera.yaml");

        slam_ = std::make_unique<ORB_SLAM3::System>(
            vocab_file, config_file, ORB_SLAM3::System::MONOCULAR, true);

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&ORBSLAM3Node::image_callback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/drone_eye/camera_pose", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/drone_eye/odom", 10);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        Sophus::SE3f pose = slam_->TrackMonocular(img, msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);

        if (!pose.matrix().isZero(0))
        {
            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = msg->header;
            pose_msg.pose.position.x = pose.translation().x();
            pose_msg.pose.position.y = pose.translation().y();
            pose_msg.pose.position.z = pose.translation().z();
            // TODO: fill orientation from Sophus::SO3

            pose_pub_->publish(pose_msg);
        }
    }

private:
    std::unique_ptr<ORB_SLAM3::System> slam_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ORBSLAM3Node>());
    rclcpp::shutdown();
    return 0;
}
