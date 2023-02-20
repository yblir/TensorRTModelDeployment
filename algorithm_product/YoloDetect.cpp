//
// Created by Administrator on 2023/2/20.
//

#include "YoloDetect.h"

extern "C" AlgorithmBase *MakeAlgorithm(void) {
    AlgorithmBase *curProduct = new YoloDetect;
    return curProduct;
}

YoloDetect::YoloDetect() = default;

//extern float d2i[6];
int YoloDetect::preProcess(cv::Mat &image, float *pinMemoryCurrentIn) {
    float d2i[6];
    std::cout<<d2i<<std::endl;
    cv::Mat scaleImage = letterBox(image, 640, 640, d2i);
    BGR2RGB(scaleImage, pinMemoryCurrentIn);
    return 0;
}

int YoloDetect::postProcess(std::vector<cv::Mat> aa,float *pinMemoryOut) {
    for(auto image:aa) {
        std::vector<std::vector<float>> boxes = decodeBox(25200, 85, pinMemoryOut, 80, 0);
        std::vector<std::vector<float>> nmsBoxes = nms(boxes, 0.25);

        //在原图上绘制检测框,并为每个框画标签
        for (auto &box: nmsBoxes)
            image = drawImage(image, box);

        cv::imwrite("image-draw3.jpg", image);
    }
}


//把不同尺度下的预测框还原到原输入图上(包括:预测框，类别概率，置信度）
std::vector<std::vector<float>>YoloDetect:: decodeBox(int boxNum, int predictNum, float *pinOutput, int classNum, const float d2i[]) {
    std:: vector<std::vector<float>> boxes;
    float confThres = 0.25;
    for (int i = 0; i < boxNum; ++i) {
        float *ptr = pinOutput + i * predictNum;
        float obj = ptr[4];
        if (obj < confThres) continue;

        float *cls = ptr + 5;
        int label = std::max_element(cls, cls + classNum) - cls;
        float predict = cls[label];
        float conf = predict * obj;
        if (conf < confThres) continue;

        //中心点,宽,高
        float cx = ptr[0], cy = ptr[1], width = ptr[2], height = ptr[3];
        //预测框
        float x1 = cx - width * 0.5, y1 = cy - height * 0.5;
        float x2 = cx + width * 0.5, y2 = cy + height * 0.5;

        //在原始图片上的位置
        float new_x1 = d2i[0] * x1 + d2i[2];
        float new_y1 = d2i[0] * y1 + d2i[5];
        float new_x2 = d2i[0] * x2 + d2i[2];
        float new_y2 = d2i[0] * y2 + d2i[5];
        boxes.push_back({new_x1, new_y1, new_x2, new_y2, (float) label, conf});
    }
    printf("decoded boxes.size = %zu\n", boxes.size());

    return boxes;
}

YoloDetect::~YoloDetect() {

}

int YoloDetect::initParam(void *param) {
    return 0;
}

int YoloDetect::postProcess(outputBase &result) {
    return 0;
}

int YoloDetect::inferImages(const std::vector<cv::Mat> &inputImages, outputBase &result) {
    return 0;
}

int YoloDetect::inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, outputBase &result) {
    return 0;
}


