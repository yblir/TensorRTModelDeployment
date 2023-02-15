//
// Created by FH on 2023/1/14.
//
#include <iostream>
#include "general.h"

//记录当前时间点
void Timer::timeStart() {
    startPoint = std::chrono::system_clock::now();
}

//记录当前时间
std::chrono::system_clock::time_point Timer::curTimePoint() {
    return std::chrono::system_clock::now();
}

// 统计当前时间与上一时间点差值
double Timer::timeCount() {
    useTime = std::chrono::system_clock::now() - startPoint;
    return useTime.count() * 1000;
}

//统计当前时间点与之前任一时间点差值
double Timer::timeCount(const std::chrono::system_clock::time_point &t1) {
    useTime = std::chrono::system_clock::now() - t1;
    return useTime.count() * 1000;
}

//遍历文件夹,返回图片名
void getImageAbsPath(const std::filesystem::path &inputDir, std::vector<std::string> &out) {
    std::filesystem::directory_iterator dirPath(inputDir);
    for (auto &it: dirPath) {
        std::string suffix = std::filesystem::path(it).extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png")
            out.push_back(it.path().string());
    }
}

//bgr转rgb,并归一化
void BGR2RGB(const cv::Mat &image, float *pinInput) {
    int imageArea = image.cols * image.rows;
    unsigned char *pImage = image.data;

    float *channelB = pinInput + imageArea * 0;
    float *channelG = pinInput + imageArea * 1;
    float *channelR = pinInput + imageArea * 2;

    for (int i = 0; i < imageArea; ++i, pImage += 3) {
        // 注意这里的顺序rgb调换了
        *channelR++ = pImage[0] / 255.0f;
        *channelG++ = pImage[1] / 255.0f;
        *channelB++ = pImage[2] / 255.0f;
//        *channelR++ = pImage[0];
//        *channelG++ = pImage[1];
//        *channelB++ = pImage[2];
    }
}