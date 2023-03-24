//
// Created by Administrator on 2023/2/20.
//

#include "YoloDetect2.h"

extern "C" AlgorithmBase *MakeAlgorithm(void) {
    AlgorithmBase *curProduct = new YoloDetect;
    return curProduct;
}

YoloDetect::YoloDetect() = default;
YoloDetect::~YoloDetect() = default;

int YoloDetect::preProcess(ParamBase &param, cv::Mat &image, float *pinMemoryCurrentIn) {
    float d2i[6];
    cv::Mat scaleImage = letterBox(image, 640, 640, d2i);
    // 依次存储一个batchSize中图片放射变换参数
    param.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});

    BGR2RGB(scaleImage, pinMemoryCurrentIn);

    return 0;
}

//cv::Mat YoloDetect::preProcess(ParamBase &param, cv::Mat &image) {
//    float d2i[6];
//    cv::Mat scaleImage = letterBox(image, 640, 640, d2i);
//    // 依次存储一个batchSize中图片放射变换参数
//    param.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
//
//    BGR2RGB(scaleImage, pinMemoryCurrentIn);
//
//    return 0;
//}

int YoloDetect::postProcess(ParamBase &param, float *pinMemoryOut, int singleOutputSize,
                            int outputNums, std::vector<std::vector<std::vector<float>>> &result) {
//std::vector<std::vector<std::vector<float>>> YoloDetect::postProcess(ParamBase &param, float *pinMemoryOut, int singleOutputSize, int outputNums, ResultBase &result) {
    //将父类对象转为子类对象,这样才能调用属于子类的成员变量
    auto curParam = reinterpret_cast<YoloDetectParam &>(param);
//    auto curResult = reinterpret_cast<YoloDetectResult &>(result);

    // outPutNums是实际推理的图片数量. 正常运行时outPutNums等于batchSize, 但在最后推理阶段, outPutNums是小于batchSzie的
    for (int i = 0; i < outputNums; ++i) {
        // 处理图片时要跳过前面已经处理的图片
        std::vector<std::vector<float>> boxes = decodeBox(curParam.predictNums, curParam.predictLength, pinMemoryOut + i * singleOutputSize,
                                                          curParam.classNums, curParam.scoreThresh, param.d2is[i]);
        std::vector<std::vector<float>> predict = nms(boxes, curParam.iouThresh);
        result.push_back(predict);
    }

    return 0;
}

//把不同尺度下的预测框还原到原输入图上(包括:预测框，类别概率，置信度),并提取得分高于阈值的预测框.
std::vector<std::vector<float>> YoloDetect::decodeBox(int predictNums, int predictLength,
                                                      float *pinOutput, int classNum, float scoreThresh, std::vector<float> d2i) {
    std::vector<std::vector<float>> boxes;

    for (int i = 0; i < predictNums; ++i) {
        float *ptr = pinOutput + i * predictLength;
        float obj = ptr[4];
        if (obj < scoreThresh) continue;

        float *cls = ptr + 5;
        int label = std::max_element(cls, cls + classNum) - cls;
        float predict = cls[label];
        float score = predict * obj;
        if (score < scoreThresh) continue;

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

        boxes.push_back({new_x1, new_y1, new_x2, new_y2, (float) label, score});
    }
//    printf("decoded boxes.size = %zu\n", boxes.size());

    return boxes;
}

int YoloDetect::initParam(void *param) {
    return 0;
}

std::vector<std::vector<std::vector<float>>> YoloDetect::getCurResult() {
    return m_curResult;
}



