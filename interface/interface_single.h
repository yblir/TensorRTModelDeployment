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
    productParam *param;
//    InputData *data;

    Infer *curAlg;
//    记录trt 输入输出需要内存大小
    float *gpuIn = nullptr, *pinMemoryIn = nullptr;
    float *gpuOut = nullptr, *pinMemoryOut = nullptr;
    unsigned long trtInSize = 0, trtOutSize = 0;
    batchBoxesType batchBox, batchBoxes;

    int singleInputSize, singleOutputSize;
//    unsigned long singleOutputSize;
    //创建cuda任务流,对应上述三个处理线程
    cudaStream_t commitStream{};
//    std::vector<cv::Mat> mats;
};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H