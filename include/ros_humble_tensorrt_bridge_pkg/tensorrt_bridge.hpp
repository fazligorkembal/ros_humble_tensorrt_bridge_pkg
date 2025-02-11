#ifndef TENSORRT_BRIDGE_HPP
#define TENSRORT_BRIDGE_HPP

#include <iostream>
#include "rclcpp/rclcpp.hpp"

#include "ros_humble_tensorrt_bridge_pkg/tensorrt/engine.h"
#include "ros_humble_tensorrt_bridge_pkg/tensorrt/logger.h"


class TensorRTBridge
{
public:
    TensorRTBridge(rclcpp::Logger logger_r, std::string model_path, std::string precision_type);
    void modelInfo();
    void predict(std::string imgPath);
    ~TensorRTBridge();

private:
    
    rclcpp::Logger logger_;
    Options options;
    Engine<float> engine;
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;
};

#endif // TENSORRT_BRIDGE_HPP