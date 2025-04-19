#include"sam2.h"

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

Encoder::~Encoder() {
    // 释放 GPU 上的 buffer
    for (void* buf : bindings) {
        if (buf) {
            cudaFree(buf);
        }
    }

    // 销毁 TensorRT 的执行上下文、引擎和 runtime
    if (context) {
        context->destroy();
        context = nullptr;
    }

    if (engine) {
        engine->destroy();
        engine = nullptr;
    }

    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
}

Decoder::~Decoder() {
    // 释放 GPU 上的 buffer
    for (void* buf : bindings) {
        if (buf) {
            cudaFree(buf);
        }
    }

    // 销毁 TensorRT 的执行上下文、引擎和 runtime
    if (context) {
        context->destroy();
        context = nullptr;
    }

    if (engine) {
        engine->destroy();
        engine = nullptr;
    }

    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
}

//input 
bool Decoder::Decoder_init(std::string path)
{   
    //load engine
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file." << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t engine_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    context = engine->createExecutionContext();
    input_size.resize(8); 
    //[1,256,64,64]  [1,32,256,256] [1,64,128,128] 

    input_size[0] = nvinfer1::Dims4{1,256,64,64}; 
    input_size[1] = nvinfer1::Dims4{1,32,256,256}; 
    input_size[2] = nvinfer1::Dims4{1,64,128,128}; 
    input_size[3] = nvinfer1::Dims3{1,2,2}; 
    input_size[4] = nvinfer1::Dims2{1,2}; 
    input_size[5] = nvinfer1::Dims4{1,1,256,256}; 
    input_size[6].nbDims = 1;
    input_size[6].d[0] = 1;
    input_size[7].nbDims = 1;
    input_size[7].d[0] = 2;

    //init input shape
    for (int i = 0; i < 8; ++i) {
        if (engine->bindingIsInput(i)) {
            if(i>2){
            context->setBindingDimensions(i, input_size[i]);
            }
        }
    }

    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "Not all dynamic input dimensions were specified!" << std::endl;
        return false;
    }

    //malloc cuda
    bindings.resize(engine->getNbBindings(), nullptr);
    buffer_sizes.resize(engine->getNbBindings(), 0);
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        size_t dtype_size = 4; // 默认 float32 = 4 bytes
        buffer_sizes[i] = size * dtype_size;
        cudaMalloc(&bindings[i], buffer_sizes[i]);

        std::cout << (engine->bindingIsInput(i) ? "[Input]" : "[Output]") 
                  << " Binding " << i << " - Size: " << buffer_sizes[i] << " bytes" << std::endl;
    }
    return true;
}

//input [1,3,1024,1024]
//output [1,256,64,64] [1,64,128,128] [1,32,256,256]
bool Encoder::Encoder_init(std::string path)
{   
    //load engine
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file." << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t engine_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    context = engine->createExecutionContext();


    //malloc cuda
    bindings.resize(engine->getNbBindings(), nullptr);
    buffer_sizes.resize(engine->getNbBindings(), 0);

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }

        size_t dtype_size = 4; // 默认 float32 = 4 bytes
        buffer_sizes[i] = size * dtype_size;
        cudaMalloc(&bindings[i], buffer_sizes[i]);

        std::cout << (engine->bindingIsInput(i) ? "[Input]" : "[Output]") 
                  << " Binding " << i << " - Size: " << buffer_sizes[i] << " bytes" << std::endl;
    }
    return true;
}

void Encoder::process(cv::Mat& input_image)
{
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(1024, 1024));

    // 2. BGR to RGB
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // 3. Convert to float32 and normalize to [0, 1]
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);

    // 4. Mean and std
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    // 5. HWC to CHW + Normalize
    std::vector<float> chw_image(3 * 1024 * 1024);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 1024; ++h) {
            for (int w = 0; w < 1024; ++w) {
                float pixel = resized_image.at<cv::Vec3f>(h, w)[c];
                chw_image[c * 1024 * 1024 + h * 1024 + w] = (pixel - mean[c]) / std[c];
            }
        }
    }
    cudaMemcpy(bindings[0], chw_image.data(), buffer_sizes[0], cudaMemcpyHostToDevice);

}

bool Encoder::infer(std::vector<void*>& output_ptrs, std::vector<size_t>& output_sizes)
{
    if (!context->enqueueV2(bindings.data(), 0, nullptr)) {
        std::cerr << "Failed to run inference." << std::endl;
        return false;
    }

    output_ptrs.clear();
    output_sizes.clear();

    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (!engine->bindingIsInput(i)) {
            output_ptrs.push_back(bindings[i]);
            output_sizes.push_back(buffer_sizes[i]);
        }
    }
    // std::vector<float> host_output(output_sizes[0] / sizeof(float)); 
    // cudaMemcpy(host_output.data(), output_ptrs[0], output_sizes[0], cudaMemcpyDeviceToHost);
    // for (int i=0;i<host_output.size();++i)
    // {
    //     std::cout<<host_output[i]<<std::endl;
    // }

    return true;
}


void Decoder::process(int ori_w,int ori_h,std::vector<float> boxes,std::vector<void*>& output_ptrs, std::vector<size_t>& output_sizes)
{
    for(int i=0;i<boxes.size();++i)
    {
        if(i%2==0){
            boxes[i]=boxes[i]/ori_w*1024;
        }
        else{
            boxes[i]=boxes[i]/ori_h*1024;
        }
    }
    std::vector<float> point_labels = { 2,3 };

    std::vector<float> mask_input(1 * 1 * 256 * 256, 0.0f);
    std::vector<float> has_mask_input = { 0.0f };
    std::vector<float> frame_size = { 1024.0, 1024.0 };

    std::vector<float> host_output(output_sizes[0] / sizeof(float)); 
    cudaMemcpy(host_output.data(), output_ptrs[0], output_sizes[0], cudaMemcpyDeviceToHost);
    for (int i=0;i<host_output.size();++i)
    {
        std::cout<<host_output[i]<<std::endl;
    }

    cudaMemcpy(bindings[0], output_ptrs[0], buffer_sizes[0], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[1], output_ptrs[2], buffer_sizes[1], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[2], output_ptrs[1], buffer_sizes[2], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[3], boxes.data(), buffer_sizes[3], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[4], point_labels.data(), buffer_sizes[4], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[5], mask_input.data(), buffer_sizes[5], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[6], has_mask_input.data(), buffer_sizes[6], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[7], frame_size.data(), buffer_sizes[7], cudaMemcpyHostToDevice);
}


bool Decoder::infer()
{
    if (!context->enqueueV2(bindings.data(), 0, nullptr)) {
        std::cerr << "Failed to run inference." << std::endl;
        return false;
    }
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        if (!engine->bindingIsInput(i)) {
            size_t element_count = buffer_sizes[i] / sizeof(float);
            std::vector<float> output(element_count);

            // 拷贝 GPU -> CPU
            cudaMemcpy(output.data(), bindings[i], buffer_sizes[i], cudaMemcpyDeviceToHost);
         
            this->output_data.push_back(std::move(output));
        }
    }
    return true;
}

cv::Mat Decoder::postprecess(int ori_w, int ori_h) {
    std::vector<float> output=this->output_data[1];
    cv::Mat image(256, 256, CV_32F, output.data());
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(ori_w, ori_h));

    cv::Mat sigmoid_output(ori_h, ori_w, CV_8U);
    for (int i = 0; i < ori_h; ++i) {
        for (int j = 0; j < ori_w; ++j) {
            float val = resized.at<float>(i, j);
            float s = sigmoid(val);
            uchar pixel = static_cast<uchar>(std::min(255.0f, std::max(0.0f, s * 255.0f)));
            sigmoid_output.at<uchar>(i, j) = pixel;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(sigmoid_output, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat black_image = cv::Mat::zeros(ori_h, ori_w, CV_8UC3);
    cv::drawContours(black_image, contours, -1, cv::Scalar(0, 0, 255), 2);
    cv::imwrite("1.jpg",black_image);
    return black_image;
}

Sam2::Sam2(std::string path1,std::string path2)
{
    this->encoder.Encoder_init(path1);
    this->decoder.Decoder_init(path2);
}


void Sam2::Sam2_encoder_process(cv::Mat& input_image)
{   
    this->ori_w=input_image.cols;
    this->ori_h=input_image.rows;
    this->encoder.process(input_image);
}

void Sam2::Sam2_encoder_infer()
{   
    this->encoder.infer(this->output_ptrs,this->output_sizes);
}

void Sam2::Sam2_decoder_process()
{
    this->decoder.process(this->ori_w,this->ori_h,this->boxes,this->output_ptrs,this->output_sizes);
}

void Sam2::set_boxes(std::vector<float> boxes)
{
    this->boxes=boxes;
}

void Sam2::Sam2_decoder_infer()
{
    this->decoder.infer();
}

void Sam2::Sam2_decoder_postprocess()
{
    this->decoder.postprecess(this->ori_w,this->ori_w);
}