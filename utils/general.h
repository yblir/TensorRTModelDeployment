//
// Created by FH on 2023/1/14.
//
//统计耗时的类
#ifndef INFERCODES_TIMER_H
#define INFERCODES_TIMER_H

#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>


#endif //INFERCODES_TIMER_H

//传入输入图片文件夹路径, 和空字符串vector,返回vector,存储文件夹中所有所有图片路径
void getImageAbsPath(const std::filesystem::path &inputDir, std::vector<std::string> &out);
//BGR->RGB, 有更简单的opencv函数可供使用,手动的转换有必要吗?
void BGR2RGB(const cv::Mat &image, float *pinInput);

//记录程序耗时
class Timer {
public:
    //开始计时
    void timeStart();

    static std::chrono::system_clock::time_point curTimePoint();

    //不加参数时,统计timeStart()开始后的耗时
    double timeCount();

    //加时间点参数, 统计与curTimePoint()对应的时间点
    double timeCount(const std::chrono::system_clock::time_point &t1);

private:
    std::chrono::system_clock::time_point startPoint;
    std::chrono::duration<double> useTime;
};

// 与tensorRT引擎有关的通用工具类
class TensorRTEngine {
public:

};