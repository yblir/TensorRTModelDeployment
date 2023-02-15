//
// Created by Administrator on 2023/2/10.
//

#include "YoloFace.h"

extern "C" AlgorithmBase *MakeAlgorithm(void){
    AlgorithmBase *curProduct=new YoloFace;
    return curProduct;
}

YoloFace::YoloFace(){

}

YoloFace::~YoloFace() {

}

int YoloFace::initParam(void *param) {
    return 0;
}

int YoloFace::preProcess(std::vector<cv::Mat > inputImages) {
    return 0;
}

int YoloFace::postProcess(outputBase &result) {
    return 0;
}

int YoloFace::inferImages(const std::vector<cv::Mat> &inputImages, outputBase &result) {
    return 0;
}

int YoloFace::inferGpuImages(const std::vector<cv::cuda::GpuMat> &inputImages, outputBase &result) {
    return 0;
}




