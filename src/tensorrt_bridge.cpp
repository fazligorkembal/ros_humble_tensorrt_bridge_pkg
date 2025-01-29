#include "ros_humble_tensorrt_bridge_pkg/tensorrt_bridge.hpp"

#include <stdio.h>
__global__ void helloFromGPU(void)
{
printf("Hello World from GPU!\n");
}


TensorRTBridge::TensorRTBridge(rclcpp::Logger logger_r, std::string model_path, std::string precision_type) : logger_(logger_r)
{
    RCLCPP_INFO(logger_, "[TensorRTBridge] - constructor\n \
        model_path: %s\n \
        precision_type: %s",
                model_path.c_str(), precision_type.c_str());

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel(logLevelStr);
    spdlog::set_level(logLevel);

    options.precision = Precision::FP16;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    engine.setOptions(options);

    bool succ = engine.buildLoadNetwork(model_path, subVals, divVals, normalize);
    if (!succ)
    {
        throw std::runtime_error("Unable to build or load TensorRT engine.");
    }
    else
    {
        RCLCPP_INFO(logger_, "[TensorRTBridge] - Engine built and loaded successfully");
    }
}

TensorRTBridge::~TensorRTBridge()
{
    for (const auto &inputDim : engine.getInputDims())
    {
        std::vector<cv::cuda::GpuMat> batchInput_d;
    }
}

void TensorRTBridge::predict(cv::Mat input_h)
{
    if (input_h.empty())
    {
        RCLCPP_ERROR(logger_, "[TensorRTBridge] - Input image is empty");
        throw std::runtime_error("Input image is empty");
    }

    cv::resize(input_h, input_h, cv::Size(640, 640));

    cv::imshow("Input", input_h);
    cv::waitKey(0);

    cv::cuda::GpuMat input_d;
    input_d.upload(input_h);
    cv::cuda::cvtColor(input_d, input_d, cv::COLOR_BGR2RGB);

    std::vector<std::vector<cv::cuda::GpuMat>> batchInput;

    size_t batchSize = options.optBatchSize;

    for (const auto &inputDim : engine.getInputDims())
    {
        // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j)
        { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination
            // of the two in order to maintain the aspect ratio You can use the
            // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
            // maintain the aspect ratio (adds padding where necessary to achieve
            // this).
            auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(input_d, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect
            // ratio with the use of padding by using the following instead:
            //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
            //            inputDim.d[1])); // TRT dims are (height, width) whereas
            //            OpenCV is (width, height)
            input.push_back(resized);
        }
        batchInput.push_back(input);
    }

    std::vector<std::vector<std::vector<float>>> featureVectors;

    bool succ = engine.runInference(batchInput, featureVectors);
    if (!succ)
    {
        const std::string msg = "Unable to run inference.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    int hold_index = -1;
    int boxes = 0;

    /*
    for (int i = 0; i < 84; i++)
    {
        for (int j = 0; j < 8400; j++)
        {
            if (featureVectors[0][0][i * 8400 + j] > 0.5)
            {
                auto xcenter = featureVectors[0][0][0 * 8400 + j];
                auto ycenter = featureVectors[0][0][1 * 8400 + j];
                auto width = featureVectors[0][0][2 * 8400 + j];
                auto height = featureVectors[0][0][3 * 8400 + j];

                for (int jj = 4; jj < 84; jj++)
                {
                    if (featureVectors[0][0][jj * 8400 + j] > 0.1)
                    {
                        std::cout << "xcenter: " << xcenter << std::endl;
                        std::cout << "ycenter: " << ycenter << std::endl;
                        std::cout << "width: " << width << std::endl;
                        std::cout << "height: " << height << std::endl;
                        std::cout << "class: " << jj - 4 << std::endl;
                        std::cout << "confidence: " << featureVectors[0][0][jj * 8400 + j] << std::endl;

                        int x1 = (xcenter - width / 2);
                        int y1 = (ycenter - height / 2);
                        int x2 = (xcenter + width / 2);
                        int y2 = (ycenter + height / 2);

                        cv::Rect rect(x1, y1, x2 - x1, y2 - y1);


                        cv::rectangle(input_h, rect, cv::Scalar(0, 255, 0), 2);
                        boxes++;
                    }
                }
            }
        }
    }

    std::cout << "boxes: " << boxes << std::endl;
    cv::imshow("Output", input_h);
    cv::waitKey(0);
    */
}

void TensorRTBridge::modelInfo()
{
    RCLCPP_INFO(logger_, "[TensorRTBridge] - modelInfo");
    RCLCPP_INFO(logger_, "[TensorRTBridge] - Engine IO Names");
    for (auto &name : engine.getIOTensorNames())
    {
        RCLCPP_INFO(logger_, "[TensorRTBridge]     %s", name.c_str());
    }

    std::vector<nvinfer1::Dims3> inputDims = engine.getInputDims();
    RCLCPP_INFO(logger_, "[TensorRTBridge] - Engine Input Dims");
    for (auto &dims : inputDims)
    {
        RCLCPP_INFO(logger_, "[TensorRTBridge]     %dx%dx%d", dims.d[0], dims.d[1], dims.d[2]);
    }

    std::vector<nvinfer1::Dims> outputDims = engine.getOutputDims();
    RCLCPP_INFO(logger_, "[TensorRTBridge] - Engine Output Dims");
    for (auto &dims : outputDims)
    {
        RCLCPP_INFO(logger_, "[TensorRTBridge]     %dx%dx%d", dims.d[0], dims.d[1], dims.d[2]);
    }
}
