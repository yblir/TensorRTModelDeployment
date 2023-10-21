//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H

#define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>

//#include "pybind11/pybind11.h"
//#include "pybind11/stl.h"
//#include "pybind11/numpy.h"

#include "../product/product.h"
#include "../product/YoloDetect.h"

class Engine {
public:
    int initEngine(ManualParam &inputParam);

    batchBoxesType inferEngine(const pybind11::array &image);
    batchBoxesType inferEngine(const std::vector<pybind11::array> &images);

//    futureBoxes inferEngine(const cv::Mat &mat);
//    futureBoxes inferEngine(const std::vector<cv::Mat> &mats);

    int releaseEngine();

private:
    productParam *param;
    InputData *data;
//    std::vector<cv::Mat> mats;
};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H