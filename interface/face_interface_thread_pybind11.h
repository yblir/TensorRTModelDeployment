//
// Created by Administrator on 2023/1/9.
//
#ifndef FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H
#define FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H

#include <tuple>

#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "../product/product.h"
#include "../base_infer/infer.h"
#include "../product/YoloDetect.h"


using Handle = void *;

//namespace py = pybind11;
class Engine {
public:
    int initEngine(ManualParam &inputParam);

//    std::map<std::string, futureBoxes> inferEngine(const InputData &data);

    std::map<std::string, futureBoxes> inferEngine(const std::string &imgPath);
    std::map<std::string, futureBoxes> inferEngine(const std::vector<std::string> &imgPaths);

    std::map<std::string, futureBoxes> inferEngine(const pybind11::array &img);
    std::map<std::string, futureBoxes> inferEngine(const std::vector<pybind11::array> &imgs);

//    std::map<std::string, futureBoxes> inferEngine(const cv::Mat &mat);
    std::map<std::string, futureBoxes> inferEngine(const std::vector<cv::Mat> &mats);

    int releaseEngine();

private:
    productParam *param;
    InputData *data;
};

#endif //FACEFEATUREDETECTOR_REBUILD_FACE_INTERFACE_H