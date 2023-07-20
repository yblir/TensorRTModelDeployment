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
//    // 通过双线性插值对图像进行resize
//    float scaleX = width / image.cols;
//    float scaleY = height / image.rows;
//
//    float scale = std::min(scaleX, scaleY);
//
//    //仿射变换的相关参数
//    float i2d[] = {
//            scale, 0, static_cast<float>((-scale * image.cols + width + scale - 1) * 0.5),
//            0, scale, static_cast<float>((-scale * image.rows + height + scale - 1) * 0.5)
//    };
//
//    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
//    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
//
//    // 计算一个反仿射变换
//    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
//
//    cv::Mat inputImage(height, width, CV_8UC3);
//
//    // 对图像做平移缩放旋转变换,可逆
//    cv::warpAffine(image, inputImage, m2x3_i2d, inputImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
//
//    return inputImage;

    // 通过双线性插值对图像进行resize
    float scaleX = width / (float) image.cols;
    float scaleY = height / (float) image.rows;

    float scale = std::min(scaleX, scaleY);
    //仿射变换的相关参数,不懂1
    float i2d[6];
    //resize图像
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * image.cols + width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * image.rows + height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    // 计算一个反仿射变换
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat inputImage(height, width, CV_8UC3);
//    std::cout<<"inputImage.size() = "<<inputImage.size()<<std::endl;
//    printf("=======================================================");
    // 对图像做平移缩放旋转变换,可逆
    cv::warpAffine(image, inputImage, m2x3_i2d, inputImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

    return inputImage;
}

// 图片填充灰度像素到模型指定输入尺寸
cv::Mat letterBox(const cv::Mat &img, int inputHeight, int inputWidth) {

    // Get current image shape [height, width]

    // Refer to https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L111

    int img_h = img.rows;
    int img_w = img.cols;

    // Compute scale ratio(new / old) and target resized shape
    float scale = std::min(inputHeight * 1.0 / img_h, inputWidth * 1.0 / img_w);
    int resize_h = int(round(img_h * scale));
    int resize_w = int(round(img_w * scale));

    // Compute padding
    int pad_h = inputHeight - resize_h;
    int pad_w = inputWidth - resize_w;

    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    // divide padding into 2 sides
    float half_h = pad_h * 1.0 / 2;
    float half_w = pad_w * 1.0 / 2;

    // Compute padding boarder
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));

    // Add border
    cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resized_img;

}

//后处理时,
cv::Mat scaleBox(const cv::Mat &img, int inputHeight, int inputWidth) {

    // Get current image shape [height, width]

    // Refer to https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L111

    int img_h = img.rows;
    int img_w = img.cols;

    // Compute scale ratio(new / old) and target resized shape
    float scale = std::min(inputHeight * 1.0 / img_h, inputWidth * 1.0 / img_w);
    int resize_h = int(round(img_h * scale));
    int resize_w = int(round(img_w * scale));

    // Compute padding
    int pad_h = inputHeight - resize_h;
    int pad_w = inputWidth - resize_w;

    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));

    // divide padding into 2 sides
    float half_h = pad_h * 1.0 / 2;
    float half_w = pad_w * 1.0 / 2;

    // Compute padding boarder
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));

    // Add border
    cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resized_img;

}

// 传入原图,绘制坐标框和标签信息后返回
cv::Mat drawImage(cv::Mat &image, const std::vector<float> &box) {
    int x1 = int(box[0]), y1 = int(box[1]), x2 = int(box[2]), y2 = int(box[3]);
    int cls = int(box[4]);
    float confidence = box[5];

    cv::Scalar color;
    //tie 相当于python 对列表/元祖的解包操作
    std::tie(color[0], color[1], color[2]) = random_color(cls);
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    //写标签
    auto name = cocoLabels[cls];
    auto caption = cv::format("%s %.2f", name, confidence);
    int textWidth = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    //画标签的小框
    cv::rectangle(image, cv::Point(x1 - 3, y1 - 33), cv::Point(x1 + textWidth, y1), color, -1);
    cv::putText(image, caption, cv::Point(x1, y1 - 5), 0, 1, cv::Scalar::all(0), 2, 16);

    return image;
}