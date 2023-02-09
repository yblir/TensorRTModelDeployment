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



