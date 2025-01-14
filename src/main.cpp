#include <iostream>

#include "ros_humble_tensorrt_bridge_pkg/tensorrt_bridge.hpp"
#include "rclcpp/rclcpp.hpp"

int main()
{
    rclcpp::init(0, nullptr);
    auto node = std::make_shared<rclcpp::Node>("tensorrt_bridge_node");
    auto logger = node->get_logger();
    TensorRTBridge tensorrt_bridge(logger, "model_path", "precision_type");
    
    return 0;
}