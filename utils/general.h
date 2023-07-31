//
// Created by FH on 2023/1/14.
//
//统计耗时的类
#ifndef INFERCODES_TIMER_H
#define INFERCODES_TIMER_H

#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "../base_infer/infer.h"
#include "../base_infer/selfDataType.hpp"

//传入输入图片文件夹路径, 和空字符串vector,返回vector,存储文件夹中所有所有图片路径
void getImageMatFromPath(const std::filesystem::path &imageDirPath, std::vector<cv::Mat> &matVector);
void getImagePath(const std::filesystem::path &imageDirPath, std::vector<std::string> &imagePaths);

// 通用函数,所有推理引擎都可使用. 传入基础参数,获得拼接后引擎文件名和输出路径, 与onnx文件在同一目录下
std::string getEnginePath(const BaseParam &param);

//加载算法so文件
Infer *loadDynamicLibrary(const std::string &soPath);

//BGR->RGB, 有更简单的opencv函数可供使用,手动的转换有必要吗?
void BGR2RGB(const cv::Mat &image, float *pinInput);

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

//记录程序耗时
class Timer {
public:
    //开始计时
    void timeStart();

    std::chrono::system_clock::time_point curTimePoint();

    //不加参数时,统计timeStart()开始后的耗时
    double timeCount();

    //加时间点参数, 统计与curTimePoint()对应的时间点,毫秒计时
    double timeCountMs(const std::chrono::system_clock::time_point &t1);
    //加时间点参数, 统计与curTimePoint()对应的时间点,秒计时
    double timeCountS(const std::chrono::system_clock::time_point &t1);
private:
    std::chrono::system_clock::time_point startPoint;
    std::chrono::duration<double> useTime;
};

const char *severity_string(nvinfer1::ILogger::Severity t);

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override;
};


#define checkRuntime(op) check_cuda_runtime((op),#op,__FILE__,__LINE__)

bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> ptrFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

//检验fp16或32 枚举类型是否合规.
const char *mode_string(Mode type);

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char *cocoLabels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
};

#endif //INFERCODES_TIMER_H