//
// Created by Administrator on 2023/2/24.
//
#include <iostream>
//// 1 包含头文件
//#include <fstream>
//#include <string>
//#include <filesystem>
#include <vector>
#include <queue>
//#include <cstring>
//#include <opencv2/opencv.hpp>
//#include "loguru.hpp"
#include "infer.h"
#include <chrono>

using namespace std;
std::mutex lock_;
std::queue<int> q_mats;
//存储与q_mats队列中对应图片的图片缩放参数
std::queue<std::vector<float>> q_d2is;

std::condition_variable condition;
//int main(int argc, char* argv[]){
//    loguru::init(argc, argv);
//    loguru::add_file("everything.log", loguru::Append, loguru::Verbosity_MAX);
//    LOG_F(INFO, "Hello log file!");
//    return 0;
//}

//// 接口类,纯虚函数
//// 原则是,只暴露调用者需要的函数,其他一概不暴露
//class InferInterface {
//public:
//    virtual void forward() = 0;
//};
//
//shared_ptr<InferInterface> create_infer(const string &file);

//class InferImpl : public InferInterface {
//public:
//    bool load_model(const string &file) {
//
//        context = file;
//    }
//
//    void forward() override {
//        //异常逻辑处理
//        if (context.empty()) {
//            return;
//        }
//
//        //正常执行逻辑
//        printf("使用%s进行推理\n", context.c_str());
//    }
//
//    void destroy() {
//        context.clear();
//    }
//
//private:
//    // 上下文管理器
//    string context;
//};
//
//// 获取infer实例,即表示加载模型
//// 加载模型失败,则表示资源获取失败,他们强绑定
//// 加载成功,则资源获取成功
//shared_ptr<InferInterface> create_infer(const string &file) {
//    shared_ptr<InferImpl> instance(new InferImpl());
//    if (!instance->load_model(file))
//        instance.reset();
//    return instance;
//}

void a1() {
    for (int i = 0; i < 10; ++i) {
        std::unique_lock<std::mutex> l(lock_);
//        std::cout << q_mats.size() << std::endl;
        condition.wait(l, [&]() {
            // false, 表示继续等待.
            // true, 表示不等待,跳出wait
            return q_mats.size() < 5;
        });
        q_mats.push(i);
        printf("--------------------------------\n");
        printf("%d\n", q_mats.size());
    }
}

void a2(int &count) {
    while (!q_mats.empty()) {
        {// 消费时加锁
            std::lock_guard<std::mutex> l(lock_);
            auto mat = q_mats.front();
            //从前面删除
            q_mats.pop();
            // 消费掉一个,就可以通知队列跳出等待,继续生产
            condition.notify_one();
//            std::cout << "current i: " << mat << std::endl;
            printf("%s mat: %d\n", "current", mat);
            this_thread::sleep_for(std::chrono::milliseconds(500));
            printf("aa %d\n", q_mats.size());
            count += 1;
        }
    }
//    count = 0;
}

void a3(int &count) {
    while (!q_mats.empty()) {
        std::unique_lock<std::mutex> l(lock_);
        condition.wait(l, [&]() {
            // false, 表示继续等待.
            // true, 表示不等待,跳出wait
            // 当gpuMemoryIn填入预处理图片数据为batchSize时,执行推理. 或者最后图片数量不够一个batch也可以推理,此时q_mats一定是空的
            return count >= 3 || q_mats.empty();
        });
        // 执行异步推理
        std::cout << "模拟推理" << std::endl;
        count = 0;
        // 消费掉一个,就可以通知队列跳出等待,继续生产
        condition.notify_all();
    }
}

int main() {
//    Infer infer;
//    infer.forward();
//    auto infer = create_infer("a");
//    if (nullptr == infer) {
//        printf("failed.\n");
//        return -1;
//    }
//
//    // 并行向队列传输任务,只在需要时才取回结果
//    auto fa = infer->forward("a");
//    auto fb = infer->forward("b");
//    auto fc = infer->forward("c");
//
//    printf("%s\n", fa.get().c_str());
//    printf("%s\n", fb.get().c_str());
//    printf("%s\n", fc.get().c_str());

    int count = 0;
    std::thread t1(a1);
    std::thread t2(a2, std::ref(count));
//    std::thread t3(a3, std::ref(count));

    t1.join();
    t2.join();
//    t3.join();

    return 0;
}