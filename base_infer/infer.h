//
// Created by Administrator on 2023/3/2.
//

#ifndef TENSORRTMODELDEPLOYMENT_INFER_H
#define TENSORRTMODELDEPLOYMENT_INFER_H

#include <iostream>
#include <vector>
#include <future>

#include <opencv2/opencv.hpp>
#include "selfDataType.hpp"

#ifdef PYBIND11
    #include <pybind11/numpy.h>
#endif



class Infer {
public:
    Infer() = default;
    virtual ~Infer() = default;

    //   仅在single_infer.cpp中有用到
    virtual std::vector<int> getMemory() {};

    virtual std::shared_future<batchBoxesType> commit(const InputData *data) {};

    virtual int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) = 0;

    virtual int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) = 0;
    virtual int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, std::map<int, imgBoxesType> &result, const int &index) = 0;

#ifdef PYBIND11
    virtual int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) = 0;
#endif

/*  弃用
    virtual batchBoxesType commit(BaseParam *param, const InputData *data) {};
    virtual batchBoxesType commit(BaseParam *param, const pybind11::array &image) {};
    virtual batchBoxesType commit(BaseParam *param, std::vector<pybind11::array> &images) {};
//    全部改为传递指针param
    virtual int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) = 0;
    virtual int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) = 0;
    virtual int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, const int &index, batchBoxesType &result) = 0;
    virtual int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) = 0;
*/
/*  弃用
    virtual int preProcess(BaseParam &param, const std::vector<pybind11::array> &images, float *pinMemoryCurrentIn) = 0;
    virtual int preProcess(BaseParam *param, const std::vector<pybind11::array> &images, float *pinMemoryCurrentIn) = 0;
*/
};

//extern "C" std::shared_ptr<Infer> createInfer(BaseParam &param, Infer &curFunc);
std::shared_ptr<Infer> createInfer(Infer &curFunc, BaseParam &param);

//typedef Infer *(*CreateAlgorithm)();

#endif //TENSORRTMODELDEPLOYMENT_INFER_H
