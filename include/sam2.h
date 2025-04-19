#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class Decoder{
    public:
    Decoder(){};
    ~Decoder();
    bool Decoder_init(std::string);
    void process(int ori_w,int ori_h,std::vector<float> boxes,std::vector<void*>& output_ptrs, std::vector<size_t>& output_sizes);
    cv::Mat postprecess(int ori_w,int ori_h);
    // cv::Mat postprocess(int ori_w,int ori_h,);
    bool infer();
    private:
    Logger logger;
    nvinfer1::IRuntime* runtime ;
    nvinfer1::ICudaEngine* engine ;
    nvinfer1::IExecutionContext* context ;
    std::vector<nvinfer1::Dims> input_size;
    std::vector<void*> bindings;
    std::vector<size_t> buffer_sizes;
    std::vector<std::vector<float>> output_data;
};

class Encoder{
    public:
    Encoder(){};
    ~Encoder();
    bool Encoder_init(std::string);
    void process(cv::Mat& input_image);
    bool infer(std::vector<void*>& output_ptrs, std::vector<size_t>& output_sizes);
    private:
    Logger logger;
    nvinfer1::IRuntime* runtime ;
    nvinfer1::ICudaEngine* engine ;
    nvinfer1::IExecutionContext* context ;
    std::vector<void*> bindings;
    std::vector<size_t> buffer_sizes;
};

class Sam2{
    public:
    Sam2(std::string encoder_path,std::string decoder_path);
    void Sam2_encoder_process(cv::Mat& input_image);
    void Sam2_encoder_infer();
    void Sam2_decoder_process();
    void set_boxes(std::vector<float> boxes);
    void Sam2_decoder_infer();
    void Sam2_decoder_postprocess();
    private:
    int ori_w,ori_h;
    std::vector<void*> output_ptrs;//encoder output
    std::vector<size_t> output_sizes;//encoder output_size
    std::vector<float> boxes;
    Encoder encoder;
    Decoder decoder;
};

