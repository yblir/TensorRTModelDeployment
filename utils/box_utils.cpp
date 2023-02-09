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
