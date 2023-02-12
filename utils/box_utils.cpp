//
// Created by FH on 2023/2/1.
//

#include "box_utils.h"

float iou(const std::vector<float> &a, const std::vector<float> &b) {
    //x1,y1,x2,y2
    float inter[] = {std::max(a[0], b[0]), std::max(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3])};

    float interArea = std::max(0.0f, inter[2] - inter[0]) * std::max(0.0f, inter[3] - inter[1]);
    float unionArea = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1])
                      + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - interArea;

    if (0 == interArea || 0 == unionArea) return 0.0f;

    return interArea / unionArea;
}

// box结构: x1,y1,x2,y2,c,conf
std::vector<std::vector<float>> nms(std::vector<std::vector<float>> &boxes, const float &nmsThreshold) {
    std::sort(boxes.begin(), boxes.end(), [](std::vector<float> &a, std::vector<float> &b) { return a[5] > b[5]; });

    //bool类型数组,初始全是0,数组长度为boxes.size()
    std::vector<bool> removeFlag(boxes.size());
    std::vector<std::vector<float>> nmsResult;
    nmsResult.reserve(boxes.size());

    for (int i = 0; i < boxes.size(); ++i) {
        if (removeFlag[i]) continue;

        auto &box_i = boxes[i];
        //效果与push_back一样,都会在尾部插入元素,但emplace不会拷贝元素,更高效
        nmsResult.emplace_back(box_i);
        // 当前框i与它后面的所有框进行比较
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (removeFlag[j]) continue;

            auto &box_j = boxes[j];
            //只对相同类别做过滤. 类型相同且二者iou
            if (box_i[4] == box_j[4] && iou(box_i, box_j) >= nmsThreshold) {
                removeFlag[j] = true;
            }
        }
    }

    return nmsResult;
}

// 在原始图片上添加灰度条并缩放输入尺寸(640,640),
cv::Mat letterBox(cv::Mat &image, const int &width, const int &height, float d2i[]) {
    // 通过双线性插值对图像进行resize
    float scaleX = width / (float) image.cols;
    float scaleY = height / (float) image.rows;

    float scale = std::min(scaleX, scaleY);

    //仿射变换的相关参数
    float i2d[] = {
            scale, 0, static_cast<float>((-scale * image.cols + width + scale - 1) * 0.5),
            0, scale, static_cast<float>((-scale * image.rows + height + scale - 1) * 0.5)
    };

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);

    // 计算一个反仿射变换
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat inputImage(height, width, CV_8UC3);

    // 对图像做平移缩放旋转变换,可逆
    cv::warpAffine(image, inputImage, m2x3_i2d, inputImage.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

    return inputImage;
}