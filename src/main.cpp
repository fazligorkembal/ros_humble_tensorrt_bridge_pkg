#include <iostream>

#include "ros_humble_tensorrt_bridge_pkg/tensorrt_bridge.hpp"
#include "rclcpp/rclcpp.hpp"

#include "ros_humble_tensorrt_bridge_pkg/kernels/deviceInfo.hpp"

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<rclcpp::Node>("tensorrt_bridge_node");
    auto logger = node->get_logger();
    TensorRTBridge tensorrt_bridge(logger, "/home/user/Documents/ros2_exp_ws/tensorrt_bridge_folder/src/yolo11n.onnx", "precision_type");
    tensorrt_bridge.modelInfo();

    cv::Mat input = cv::imread("/home/user/Documents/ros2_exp_ws/test.jpg");
    tensorrt_bridge.predict(input);

    who_am_i_wrapper();

    return 0;
}