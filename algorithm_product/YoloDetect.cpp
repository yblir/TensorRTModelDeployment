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
int YoloDetect::preProcess(cv::Mat &image, float *pinMemoryCurrentIn, parmBase parm) {
    cv::Mat scaleImage = letterBox(image, 640, 640);
    BGR2RGB(scaleImage, pinMemoryCurrentIn);
    return 0;
}

//1.6875,-0,-134.65625,-0,1.6875,0.34375
int YoloDetect::postProcess(std::vector<cv::Mat> aa, float *pinMemoryOut, parmBase parm) {
    for (auto image: aa) {
        std::vector<std::vector<float>> boxes = decodeBox(25200, 85, pinMemoryOut, 80, parm.d2i);
        std::vector<std::vector<float>> predictBoxes = nms(boxes, 0.25);

        //在原图上绘制检测框,并为每个框画标签
        for (auto &box: predictBoxes)
            image = drawImage(image, box);

        cv::imwrite("image-draw3.jpg", image);
        printf("2222222222222222\n");
    }
    return 0;
}


//把不同尺度下的预测框还原到原输入图上(包括:预测框，类别概率，置信度）
std::vector<std::vector<float>> YoloDetect::decodeBox(int boxNum, int predictNum, float *pinOutput, int classNum, const float d2i[]) {
    std::vector<std::vector<float>> boxes;
    float parmThres = 0.25;
    float d2i2[] = {1.6875, -0, -134.65625, -0, 1.6875, 0.34375};
    for (int i = 0; i < boxNum; ++i) {
        float *ptr = pinOutput + i * predictNum;
        float obj = ptr[4];
        if (obj < parmThres) continue;

        float *cls = ptr + 5;
        int label = std::max_element(cls, cls + classNum) - cls;
        float predict = cls[label];
        float parm = predict * obj;
        if (parm < parmThres) continue;

        //中心点,宽,高
        float cx = ptr[0], cy = ptr[1], width = ptr[2], height = ptr[3];
        //预测框
        float x1 = cx - width * 0.5, y1 = cy - height * 0.5;
        float x2 = cx + width * 0.5, y2 = cy + height * 0.5;

        //在原始图片上的位置
        float new_x1 = d2i2[0] * x1 + d2i2[2];
        float new_y1 = d2i2[0] * y1 + d2i2[5];
        float new_x2 = d2i2[0] * x2 + d2i2[2];
        float new_y2 = d2i2[0] * y2 + d2i2[5];

//        float ratio = std::min(inputWidth * 1.0f / imageWidth, inputHeight * 1.0f / imageHeight);
//        int newHeight = int(imageHeight * ratio);
//        int newWidth = int(imageWidth * ratio);
//
////    这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
//        float offset_w = (inputWidth - newWidth) / 2.0f / inputWidth;
//        float offset_h = (inputHeight - newHeight) / 2.0f / inputHeight;
//
////    将box_wh相对于input_image的比例修改为相对于resize_image的比例
//        float scale_x = inputHeight * 1.0f / newHeight;
//        float scale_y = inputWidth * 1.0f / newWidth;
//
//        int box_x = (cx - offset_w) / scale_x;
//        int box_y = (cy - offset_h) / scale_y;
//
//        int box_h = height * scale_y;
//        int box_w = width * scale_x;
//
//        // 左上,右下角坐标值
//        int x1 = box_x - box_w / 2;
//        int y1 = box_y - box_h / 2;
//        int x2 = box_x + box_w / 2;
//        int y2 = box_y + box_h / 2;

        boxes.push_back({new_x1, new_y1, new_x2, new_y2, (float) label, parm});
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

//int YoloDetect::preProcess(cv::Mat &image, float *pinMemoryCurrentIn, struct parmBase parm) {
////    printf("11111111111\n");
//    for (int i = 0; i < 6; ++i) {
//        std::cout << *(parm.d2i + i) << " " << std::endl;
//    }
//    cv::Mat scaleImage = letterBox(image, 640, 640, parm.d2i);
//    BGR2RGB(scaleImage, pinMemoryCurrentIn);
//    return 0;
//    return 0;
//}


