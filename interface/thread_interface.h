//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H

// todo 必须有这行宏定义,才能跳过python gil锁的检测, 目前没发现问题.
// todo 有了这行宏定义, pybind11::call_guard<pybind11::gil_scoped_release>()在m.def()中有没有都可以, 不懂背后运行原理.
#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "../product/product.h"
#include "../product/YoloDetect.h"

class Engine {
public:
    int initEngine(ManualParam &inputParam);

//    futureBoxes inferEngine(const InputData &data);


    futureBoxes inferEngine(const pybind11::array &img);
    futureBoxes inferEngine(const std::vector<pybind11::array> &imgs);

    futureBoxes inferEngine(const cv::Mat &mat);
    futureBoxes inferEngine(const std::vector<cv::Mat> &mats);

    int releaseEngine();

private:
//    productParam *param;
    Infer *curAlg;
    YoloDetectParam *curAlgParam;
    InputData *data;
//    std::vector<cv::Mat> mats;
};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H