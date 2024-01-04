//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H

#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "../base_infer/infer.h"
#include "../product/product.h"
#include "../product/YoloDetect.h"
//#include "../taskflow/taskflow.hpp"
// thread_pool经过国内某个大佬改写, 这里又改些了addTread函数
#include "../utils/thread_pool.hpp"
//thread_pool2是原版, 但速度比较慢? 没测出多大差别,不过代码可读性较差
//#include "../utils/thread_pool2.hpp"

class Engine {
public:
    int initEngine(ManualParam &inputParam);
    ~Engine();
    batchBoxesType inferEngine(const pybind11::array &image);
    batchBoxesType inferEngine(const std::vector<pybind11::array> &images);

    batchBoxesType inferEngine(const cv::Mat &mat);
    batchBoxesType inferEngine(const std::vector<cv::Mat> &mats);

    int releaseEngine();

private:
//    productParam *curAlgParam;
    Infer *curAlg;
    YoloDetectParam *curAlgParam;
//    BaseParam *curAlgParam;

//    记录trt 输入输出需要内存大小
    float *gpuIn = nullptr, *pinMemoryIn = nullptr;
    float *gpuOut = nullptr, *pinMemoryOut = nullptr;
    unsigned long trtInMemorySize = 0, trtOutMemorySize = 0;
    batchBoxesType batchBox, batchBoxes;
//    多线程后处理, 由于每张图片输出框数量不定, 不能统一初始化vector, 所以使用字典作为中转存储空间
    std::map<int, imgBoxesType> boxDict;

    int singleInputSize, singleOutputSize;
//    unsigned long singleOutputSize;
    //创建cuda任务流,对应上述三个处理线程
    cudaStream_t commitStream{};
//    std::vector<cv::Mat> mats;

    // Create a taskflow object
//    tf::Taskflow taskflow;

    // Create a taskflow executor
//    tf::Executor executor;

    // Create a vector of tasks for image copying
//    std::vector<tf::Task> tasks;
//    std::vector<std::thread> threads;

    std::ThreadPool preExecutor;
    std::ThreadPool postExecutor;
//    判断当前线程是否完成
    std::vector<std::future<void> > preThreadFlags;
    std::vector<std::future<void>> postThreadFlags;

};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H