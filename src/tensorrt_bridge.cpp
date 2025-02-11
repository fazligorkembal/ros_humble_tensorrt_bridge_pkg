#include "ros_humble_tensorrt_bridge_pkg/tensorrt_bridge.hpp"

#include <stdio.h>

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

void TensorRTBridge::predict(std::string imgPath)
{
    auto cpuImg = cv::imread(imgPath);
    if (cpuImg.empty())
    {
        const std::string msg = "Unable to read image at path: " + imgPath;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects RGB input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // In the following section we populate the input vectors to later pass for
    // inference
    const auto &inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the
    // inputs You should populate your inputs appropriately.
    for (const auto &inputDim : inputDims)
    { // For each of the model inputs...
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j)
        { // For each element we want to add to the batch...
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination
            // of the two in order to maintain the aspect ratio You can use the
            // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
            // maintain the aspect ratio (adds padding where necessary to achieve
            // this).
            auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            // You could also perform a resize operation without maintaining aspect
            // ratio with the use of padding by using the following instead:
            //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
            //            inputDim.d[1])); // TRT dims are (height, width) whereas
            //            OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    spdlog::info("Warming up the network...");
    std::vector<std::vector<std::vector<float>>> featureVectors;
    for (int i = 0; i < 100; ++i) {
        bool succ = engine.runInference(inputs, featureVectors);
        if (!succ) {
            const std::string msg = "Unable to run inference.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    size_t numIterations = 100;
    spdlog::info("Running benchmarks ({} iterations)...", numIterations);
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIterations; ++i)
    {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors);
    }
    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

    spdlog::info("Benchmarking complete!");
    spdlog::info("======================");
    spdlog::info("Avg time per sample: ");
    spdlog::info("Avg time per sample: {} ms", avgElapsedTimeMs);
    spdlog::info("Batch size: {}", inputs[0].size());
    spdlog::info("Avg FPS: {} fps", static_cast<int>(1000 / avgElapsedTimeMs));
    spdlog::info("featureVectors.size()" + std::to_string(featureVectors.size()));
    spdlog::info("======================\n");

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch)
    {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum)
        {
            spdlog::info("Batch {}, output {}", batch, outputNum);
            std::string output;
            int i = 0;
            for (const auto &e : featureVectors[batch][outputNum])
            {
                output += std::to_string(e) + " ";
                if (++i == 15)
                {
                    output += "...";
                    break;
                }
            }
            spdlog::info("{}", output);
        }
    }
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
