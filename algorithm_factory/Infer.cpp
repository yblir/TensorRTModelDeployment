//
// Created by Administrator on 2023/3/2.
//

#include "Infer.h"

class InferImpl : public Infer {
    std::shared_ptr<std::string> commit(const std::string& input) override;
};

//shared_future<string> commit(const std::vector<std::string>& imgPaths) override{
///*
//建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
//    commit 函数 https://v.douyin.com/NfJvHxm/
// */
//Job job;
//job.input = input;
//job.pro.reset(new promise<string>());
//
//shared_future<string> fut = job.pro->get_future();
//{
//lock_guard<mutex> l(lock_);
//jobs_.emplace(std::move(job));
//}
//cv_.notify_one();
//return fut;
//}