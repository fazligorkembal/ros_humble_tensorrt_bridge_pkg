#include "ros_humble_tensorrt_bridge_pkg/tensorrt_bridge.hpp"

TensorRTBridge::TensorRTBridge(rclcpp::Logger logger, std::string model_path, std::string precision_type) : logger_(logger)
{
    RCLCPP_INFO(logger_, "[TensorRTBridge] - constructor\n \
        model_path: %s\n \
        precision_type: %s",
                model_path.c_str(), precision_type.c_str());
}

TensorRTBridge::~TensorRTBridge()
{
    RCLCPP_INFO(logger_, "[TensorRTBridge] - destructor");
}

void TensorRTBridge::modelInfo()
{
    RCLCPP_INFO(logger_, "[TensorRTBridge] - modelInfo");
}
