//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H

// todo 必须有这行宏定义,才能跳过python gil锁的检测, 目前没发现问题.
// todo 有了这行宏定义, pybind11::call_guard<pybind11::gil_scoped_release>()在m.def()中有没有都可以
#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF

// #include <opencv2/opencv.hpp>

#ifdef PYBIND11
    #include "pybind11/pybind11.h"
    #include "pybind11/stl.h"
    #include "pybind11/numpy.h"
#endif

#include "../product/product.h"
#include "../product/YoloDetect.h"

class Engine {
public:
    int initEngine(const ManualParam &inputParam);

#ifdef PYBIND11
    futureBoxes inferEngine(const pybind11::array &img);
    futureBoxes inferEngine(const std::vector<pybind11::array> &imgs);
#endif

    // nodiscard: 函数在调用时，返回值必须有变量接受，不如会抛出warning
    [[nodiscard]] futureBoxes inferEngine(const cv::Mat &mat) const;
    [[nodiscard]] futureBoxes inferEngine(const std::vector<cv::Mat> &mats) const;

    void releaseEngine() const;

private:
    Infer * curAlg = nullptr;
    YoloDetectParam * curAlgParam = nullptr;
    InputData * data = nullptr;

};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H