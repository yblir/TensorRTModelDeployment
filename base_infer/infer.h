//
// Created by Administrator on 2023/3/2.
//

#ifndef TENSORRTMODELDEPLOYMENT_INFER_H
#define TENSORRTMODELDEPLOYMENT_INFER_H

#include <iostream>
#include <vector>
#include <future>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>

#include "selfDataType.hpp"

using batchBoxesType = std::vector<std::vector<std::vector<float>>>;


class Infer {
public:
    Infer() = default;
    virtual ~Infer() = default;

//    virtual std::vector<std::shared_future<std::string>> commit(const std::string &imagePath) {};
//    virtual std::shared_future<batchBoxesType> commit(const std::string &imagePath) {};
//    virtual std::shared_future<batchBoxesType> commit(const cv::Mat &mat) {};
//    virtual std::shared_future<batchBoxesType> commit(const cv::cuda::GpuMat &mat) {};
//
//    virtual std::shared_future<batchBoxesType> commit(const std::vector<std::string> &imagePaths) {};
//    virtual std::shared_future<batchBoxesType> commit(const std::vector<cv::Mat> &mats) {};
//    virtual std::shared_future<batchBoxesType> commit(const std::vector<cv::cuda::GpuMat> &mats) {};

//   仅在single_infer.cpp中有用到
    virtual std::vector<int> getMemory() {};

    virtual std::shared_future<batchBoxesType> commit(const InputData *data) {};

    virtual batchBoxesType commit(BaseParam *param, const InputData *data) {};

    virtual batchBoxesType commit(BaseParam *param, const pybind11::array &image) {};

    virtual batchBoxesType commit(BaseParam *param, std::vector<pybind11::array> &images) {};

    virtual int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam &param, const std::vector<pybind11::array> &images, float *pinMemoryCurrentIn) = 0;
    virtual int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) = 0;
};


//extern "C" std::shared_ptr<Infer> createInfer(BaseParam &param, Infer &curFunc);
std::shared_ptr<Infer> createInfer(BaseParam &param, Infer &curFunc);

typedef Infer *(*CreateAlgorithm)();

#endif //TENSORRTMODELDEPLOYMENT_INFER_H
