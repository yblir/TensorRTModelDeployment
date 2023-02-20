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

//遍历文件夹,返回图片矩阵vector
void getImageMatFromPath(const std::filesystem::path &imageDirPath, std::vector<cv::Mat> &matVector) {
    // 只遍历当前文件夹下的第一层结构
    std::filesystem::directory_iterator dirPath(imageDirPath);
    for (auto &it: dirPath) {
        std::string suffix = std::filesystem::path(it).extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png")
            matVector.emplace_back(cv::imread(it.path().string()));
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

// hsv转bgr 用得到吗?
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
    float h_plane = ((((unsigned int) id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int) id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}