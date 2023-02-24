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
int YoloDetect::preProcess(ParmBase &parm, cv::Mat &image, float *pinMemoryCurrentIn) {
    float d2i[6];
    cv::Mat scaleImage = letterBox(image, 640, 640, d2i);
    // 依次存储一个batchSize中图片放射变换参数
    parm.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});

    BGR2RGB(scaleImage, pinMemoryCurrentIn);

    return 0;
}

int YoloDetect::postProcess(ParmBase &parm, std::vector<cv::Mat> &images,
                            float *pinMemoryOut, int singleOutputSize, ResultBase &result) {
    auto yoloDetect = reinterpret_cast<YoloDetectResult& >(result);
    int i = 0;
    for (auto image: images) {
//        // 将尺寸缩放参数从二维vector提取放到float数组中, 因为仿射变换
//        std::copy(parm.d2is[i].begin(), parm.d2is[i].end(), d2i);
        std::vector<std::vector<float>> boxes = decodeBox(25200, 85, pinMemoryOut + i * singleOutputSize, 80, parm.d2is[i]);
        std::vector<std::vector<float>> predict = nms(boxes, 0.25);
        yoloDetect.result.push_back(predict);
        i += 1;
//        //在原图上绘制检测框,并为每个框画标签
//        for (auto &box: predictBoxes)
//            image = drawImage(image, box);
//        std::string imgName = "image-draw" + std::to_string(i) + ".jpg";
//        cv::imwrite(imgName, image);
    }
    return 0;
}


//把不同尺度下的预测框还原到原输入图上(包括:预测框，类别概率，置信度）
std::vector<std::vector<float>> YoloDetect::decodeBox(int boxNum, int predictNum, float *pinOutput, int classNum, std::vector<float> d2i) {
    std::vector<std::vector<float>> boxes;
    float parmThres = 0.25;

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
        float new_x1 = d2i[0] * x1 + d2i[2];
        float new_y1 = d2i[0] * y1 + d2i[5];
        float new_x2 = d2i[0] * x2 + d2i[2];
        float new_y2 = d2i[0] * y2 + d2i[5];

        boxes.push_back({new_x1, new_y1, new_x2, new_y2, (float) label, parm});
    }
//    printf("decoded boxes.size = %zu\n", boxes.size());

    return boxes;
}

YoloDetect::~YoloDetect() {

}

int YoloDetect::initParam(void *param) {
    return 0;
}

//int YoloDetect::postProcess(ResultBase &result) {
//    return 0;
//}
//
//int YoloDetect::inferImages(const std::vector<cv::Mat> &inputImages, ResultBase &result) {
//    return 0;
//}
//
//int YoloDetect::inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, ResultBase &result) {
//    return 0;
//}

//int YoloDetect::preProcess(cv::Mat &image, float *pinMemoryCurrentIn, struct ParmBase parm) {
////    printf("11111111111\n");
//    for (int i = 0; i < 6; ++i) {
//        std::cout << *(parm.d2i + i) << " " << std::endl;
//    }
//    cv::Mat scaleImage = letterBox(image, 640, 640, parm.d2i);
//    BGR2RGB(scaleImage, pinMemoryCurrentIn);
//    return 0;
//    return 0;
//}


