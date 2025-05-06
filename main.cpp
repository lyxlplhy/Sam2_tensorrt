#include"sam2.h"
#include <chrono>
int main()
{   std::string encoder_path="/gemini/code/Sam2-collection/sam2_tensorrt/tensorrt/conver_tiny_encoder.engine";
    std::string decoder_path="/gemini/code/Sam2-collection/sam2_tensorrt/tensorrt/conver_tiny_decoder.engine";
    std::string img_path="/home/sam2TRT-main/test_images/00001896_151878_1292_724_1245_1236_0.996929.jpg";
    cv::Mat img=cv::imread(img_path);
    std::vector<std::vector<float>> a={{1400,830,2467,1891}};
    Sam2 sam2(encoder_path,decoder_path);
    auto start1 = std::chrono::high_resolution_clock::now();
    while(1){
         auto start5 = std::chrono::high_resolution_clock::now();
        sam2.sam2_infer(img,a);
         auto end5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(end5 - start5).count();
    std::cout << "总时间 " << duration5 << " ms" << std::endl;
        }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout<<"总体开销时间"<<duration1<<"ms"<<std::endl;
    std::vector<cv::Mat>masks=sam2.sam2_getmasks();
    // cv::imwrite("1.jpg",masks[0]);
    cv::imwrite("2.jpg",masks[0]);
    return 0;
}