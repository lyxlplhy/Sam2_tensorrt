#include"sam2.h"
#include <cuda_runtime.h>     // cudaMalloc, cudaMemcpy 等
#include <device_launch_parameters.h> // blockIdx/threadIdx 定义


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
    this->blob = cv::dnn::blobFromImage(input_image, 1 / 255.0, {1024,1024}, cv::Scalar(0,0,0), true, false,CV_32F);
    float mean[3] = { 0.485, 0.456, 0.406 };
    float std[3] = { 0.229, 0.224, 0.225 };
    for (int c = 0; c < 3; ++c) {
    cv::Mat channel(this->blob.size[2], this->blob.size[3], CV_32F, this->blob.ptr(0, c)); // 提取第 c 通道
    channel = (channel - mean[c]) / std[c]; // 合并减去均值和除以标准差的步骤
    }
    cudaMemcpy(bindings[0],  this->blob.ptr<float>(), buffer_sizes[0], cudaMemcpyHostToDevice);
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
    return true;
}


void Decoder::process(int ori_w,int ori_h,std::vector<float> boxes,std::vector<void*>& output_ptrs, std::vector<size_t>& output_sizes,bool muti)
{
    if(muti)
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
    cudaMemcpy(bindings[0], output_ptrs[0], buffer_sizes[0], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[1], output_ptrs[2], buffer_sizes[1], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[2], output_ptrs[1], buffer_sizes[2], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[3], boxes.data(), buffer_sizes[3], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[4], point_labels.data(), buffer_sizes[4], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[5], mask_input.data(), buffer_sizes[5], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[6], has_mask_input.data(), buffer_sizes[6], cudaMemcpyHostToDevice);
    cudaMemcpy(bindings[7], frame_size.data(), buffer_sizes[7], cudaMemcpyHostToDevice);
    }
    else{
        for(int i=0;i<boxes.size();++i)
    {
        if(i%2==0){
            boxes[i]=boxes[i]/ori_w*1024;
        }
        else{
            boxes[i]=boxes[i]/ori_h*1024;
        }
    }
        cudaMemcpy(bindings[3], boxes.data(), buffer_sizes[3], cudaMemcpyHostToDevice);
    }
}


bool Decoder::infer()
{
    this->output_data.clear();
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

cv::Mat Decoder::postprecess(int &ori_w, int &ori_h) {
   
    const std::vector<float>& output = this->output_data[1];
    cv::Mat image(256, 256, CV_32F, const_cast<float*>(output.data()));
    cv::resize(image, resized, cv::Size(ori_w, ori_h));
    resized.convertTo(binary_mask, CV_8U, 255.0);  // scale float to [0,255]   
    cv::threshold(binary_mask, binary_mask, 0, 255, cv::THRESH_BINARY); 
   
    return binary_mask;
}

Sam2::Sam2(std::string path1,std::string path2)
{
    this->encoder.Encoder_init(path1);
    this->decoder.Decoder_init(path2);
    this->img_masks.reserve(5);  
}

void Sam2::sam2_infer(cv::Mat& img,std::vector<std::vector<float>>& boxes)
{   this->boxes.clear();
    this->boxes=boxes;
    auto start1 = std::chrono::high_resolution_clock::now();
    this->output_ptrs.clear();
    this->output_sizes.clear();
    this->img_masks.clear();
    this->ori_w=img.cols;
    this->ori_h=img.rows;

    this->encoder.process(img);
 
    this->encoder.infer(this->output_ptrs,this->output_sizes);
 
    bool muti=true;
    for(int i=0;i<this->boxes.size();++i)
    {
        this->decoder.process(this->ori_w,this->ori_h,this->boxes[i],this->output_ptrs,this->output_sizes,muti);
        this->decoder.infer();
        this->img_masks.push_back(this->decoder.postprecess(this->ori_w,this->ori_h));
        muti=false;
    }
}

std::vector<cv::Mat> Sam2::sam2_getmasks()
{
    return this->img_masks;
}