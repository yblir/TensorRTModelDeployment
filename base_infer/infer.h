//
// Created by Administrator on 2023/3/2.
//

#ifndef TENSORRTMODELDEPLOYMENT_INFER_H
#define TENSORRTMODELDEPLOYMENT_INFER_H

#include <iostream>
#include <vector>
#include <future>
//#include <NvInfer.h>
#include <opencv2/opencv.hpp>
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

    virtual std::shared_future<batchBoxesType> commit(const InputData *data) {};

    virtual int preProcess(BaseParam &param, cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    virtual int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) = 0;
};


//extern "C" std::shared_ptr<Infer> createInfer(BaseParam &param, Infer &curFunc);
std::shared_ptr<Infer> createInfer(BaseParam &param, Infer &curFunc);

typedef Infer *(*CreateAlgorithm)();

#endif //TENSORRTMODELDEPLOYMENT_INFER_H
