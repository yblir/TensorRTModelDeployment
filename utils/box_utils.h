//
// Created by FH on 2023/2/1.
//
//所有坐标框相关的工具函数
#ifndef INFERCODES_BOX_UTILS_H
#define INFERCODES_BOX_UTILS_H

#include <iostream>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
//#include <unistd.h>
//#include <memory>
#include <iostream>
#include "../utils/general.h"

//计算两个框直接的iou值 box结构: x1,y1,x2,y2,
float iou(const std::vector<float> &a, const std::vector<float> &b);

//计算所有预测框nms, box结构: x1,y1,x2,y2,c,conf
std::vector<std::vector<float>> nms(std::vector<std::vector<float>> &boxes, const float &nmsThreshold);

//传入原始图片,返回被修改的图片. 放射变换的方法需要额外传入变量,靠谱吗?
cv::Mat letterBox(cv::Mat &image, const int &width, const int &height, float d2i[]);
cv::Mat letterBox(const cv::Mat &img, int inputHeight, int inputWidth);

//画框
cv::Mat drawImage(cv::Mat &image, const std::vector<float> &box);


#endif //INFERCODES_BOX_UTILS_H