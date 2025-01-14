#ifndef TENSORRT_BRIDGE_HPP
#define TENSRORT_BRIDGE_HPP

#include <iostream>
#include "rclcpp/rclcpp.hpp"

class TensorRTBridge
{
public:
    TensorRTBridge(rclcpp::Logger logger, std::string model_path, std::string precision_type);
    ~TensorRTBridge();

private:
    void modelInfo();
    rclcpp::Logger logger_;
};

#endif // TENSORRT_BRIDGE_HPP