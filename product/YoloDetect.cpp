//
// Created by Administrator on 2023/2/20.
//

#include "YoloDetect.h"

//extern "C" Infer *MakeAlgorithm(void) {
//    Infer *curProduct = new YoloDetect;
//    return curProduct;
//}

YoloDetect::YoloDetect() = default;
YoloDetect::~YoloDetect() = default;
/*
int YoloDetect::preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn) {
    cv::Mat mat(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));
    return preProcess(param, mat, pinMemoryCurrentIn);

//    return 0;
}

int YoloDetect::preProcess(BaseParam &param, const std::vector<pybind11::array> &image, float *pinMemoryCurrentIn) {
    return 0;
}

int YoloDetect::preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn) {
    cv::Mat mat(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));

    return preProcess(param, mat, pinMemoryCurrentIn);

//    std::cout << "pinMemoryCurrentIn=" << *pinMemoryCurrentIn << std::endl;
//    pinMemoryCurrentIn = (float *) image.data(0);
//    std::cout << "pinMemoryCurrentIn=" << *pinMemoryCurrentIn << std::endl;
//    auto a = image.data(0);
//    return 0;
}

int YoloDetect::preProcess(BaseParam *param, const std::vector<pybind11::array> &image, float *pinMemoryCurrentIn) {
    return 0;
}
*/
int YoloDetect::preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) {

    cv::Mat scaleImage = letterBox(image, param.inputWidth, param.inputHeight, param.d2i);

    BGR2RGB(scaleImage, pinMemoryCurrentIn);
    // 依次存储一个batchSize中图片放射变换参数
    param.d2is.push_back(
            {param.d2i[0], param.d2i[1], param.d2i[2],
             param.d2i[3], param.d2i[4], param.d2i[5]}
    );
    return 0;
}

int YoloDetect::preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn) {

    cv::Mat scaleImage = letterBox(image, param->inputWidth, param->inputHeight, param->d2i);

    BGR2RGB(scaleImage, pinMemoryCurrentIn);
    // 依次存储一个batchSize中图片放射变换参数
    param->d2is.push_back(
            {param->d2i[0], param->d2i[1], param->d2i[2],
             param->d2i[3], param->d2i[4], param->d2i[5]}
    );
    return 0;
}

int YoloDetect::postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) {
    //将父类对象转为子类对象,这样才能调用属于子类的成员变量
    auto curParam = reinterpret_cast<YoloDetectParam &>(param);
    std::vector<std::vector<float>> boxes, predict;
    // outPutNums是实际推理的图片数量. 正常运行时outPutNums等于batchSize, 但在最后推理阶段, outPutNums是小于batchSize的
    for (int i = 0; i < outputNums; ++i) {
        // 处理图片时要跳过前面已经处理的图片
        boxes = decodeBox(curParam.trtOutputShape.d[1], curParam.trtOutputShape.d[2],
                          pinMemoryCurrentOut + i * singleOutputSize, curParam.classNums, curParam.scoreThresh, param.d2is[i]);
//        boxes = decodeBox(curParam.trtOutputShape.d[1], curParam.trtOutputShape.d[2],
//                          pinMemoryCurrentOut + i * singleOutputSize, curParam.classNums, curParam.scoreThresh);
        predict = nms(boxes, curParam.iouThresh);

        result.push_back(predict);
    }
    param.d2is.clear();
    return 0;
}

int YoloDetect::postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) {
    //将父类对象转为子类对象,这样才能调用属于子类的成员变量
    auto curParam = reinterpret_cast<YoloDetectParam *>(param);
    std::vector<std::vector<float>> boxes, predict;
    // outPutNums是实际推理的图片数量. 正常运行时outPutNums等于batchSize, 但在最后推理阶段, outPutNums是小于batchSize的
    for (int i = 0; i < outputNums; ++i) {
        // 处理图片时要跳过前面已经处理的图片
        boxes = decodeBox(curParam->trtOutputShape.d[1], curParam->trtOutputShape.d[2],
                          pinMemoryCurrentOut + i * singleOutputSize, curParam->classNums, curParam->scoreThresh, param->d2is[i]);
//        boxes = decodeBox(curParam->trtOutputShape.d[1], curParam->trtOutputShape.d[2],
//                          pinMemoryCurrentOut + i * singleOutputSize, curParam->classNums, curParam->scoreThresh);
        predict = nms(boxes, curParam->iouThresh);
        result.push_back(predict);
    }
//    在当前推理后处理时清空仿射变换参数,使得所有其他模型可以统一调用接口
//  不论单张还是多张推理, 只要在preprocess中设置了仿射变换,必须在postprocess中clear(), 否则后续图片检测框会错位.
    param->d2is.clear();
    return 0;
}

//把不同尺度下的预测框还原到原输入图上(包括:预测框，类别概率，置信度),并提取得分高于阈值的预测框.
std::vector<std::vector<float>> YoloDetect::decodeBox(int predictNums, int predictLength,
                                                      float *pinOutput, int classNum, float scoreThresh, std::vector<float> d2i) {
//std::vector<std::vector<float>> YoloDetect::decodeBox(int predictNums, int predictLength,
//                                                      float *pinOutput, int classNum, float scoreThresh) {
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
//        boxes.push_back({x1, y1, x2, y2, (float) label, score});
    }

    return boxes;
}


