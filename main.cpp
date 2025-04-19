#include"sam2.h"
int main()
{   std::string encoder_path="/gemini/code/Sam2-collection/sam2_tensorrt/tensorrt/conver_tiny_encoder.engine";
    std::string decoder_path="/gemini/code/Sam2-collection/sam2_tensorrt/tensorrt/conver_tiny_decoder.engine";
    std::string img_path="/home/sam2TRT-main/test_images/00001896_151878_1292_724_1245_1236_0.996929.jpg";
    cv::Mat img=cv::imread(img_path);

    Sam2 sam2(encoder_path,decoder_path);
    sam2.Sam2_encoder_process(img);
    sam2.Sam2_encoder_infer();

    sam2.set_boxes({1400,830,2467,1891});
    sam2.Sam2_decoder_process();
    sam2.Sam2_decoder_infer();
    sam2.Sam2_decoder_postprocess();
    return 0;

}