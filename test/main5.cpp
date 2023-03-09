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
#include <future>

using namespace std;
std::mutex lock_;
std::queue<int> q_mats;
//存储与q_mats队列中对应图片的图片缩放参数
std::queue<std::vector<float>> q_d2is;
bool work_running = true;

// 图片循环完毕
bool fetch_over = false;
bool transfer_finish = false;
// 为true时,可以把数据从gpu拷贝到推理空间
bool copy_over = false;

std::shared_ptr<int> trand;
std::shared_future<int> *aa;

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

void a1(bool &fetched_over) {
    for (int i = 1; i < 16; ++i) {
        {
            std::unique_lock<std::mutex> l1(lock_);
            condition.wait(l1, [&]() {
                // false, 表示继续等待.
                // true, 表示不等待,跳出wait
                return q_mats.size() < 5;
            });
            q_mats.push(i);
            this_thread::sleep_for(std::chrono::milliseconds(1));
            if (i == 15) {
                printf("a1 overrrrrrrrrrrrrrrrrrrrrrrrrr\n");
                fetched_over = true;
            }

            condition.notify_one();
        }
        printf("------------------- %lu\n", q_mats.size());
    }
    printf("a1 fetched over\n");
}

void a2(int &count, bool &fetched_over, bool &transfer_finished) {
//    while (true) {
//        // 每次向推理空间gpuMemoryIn复制batchSize个处理好的图片
//        {   // 消费时加锁
//            std::unique_lock<std::mutex> l2(lock_);
//
////          当推理空间已保存图片数量为batchSize时, 等待, 等待推理线程把这批数据取走推理,然后再重新复制
//            condition.wait(l2, [&]() {
//                // false, 表示继续等待. true, 表示不等待,跳出wait
//                // 当gpuMemoryIn填入预处理图片数据为batchSize时,执行推理. 或者最后图片数量不够一个batch也可以推理,此时q_mats一定是空的
//                return !copy_over && !q_mats.empty();
//            });
//            auto mat = q_mats.front();
//            //从前面删除
//            q_mats.pop();
//            // 消费掉一个,就可以通知队列跳出等待,继续生产
//            condition.notify_one();
//            printf("-------------------> %s mat: %d, q_mat.size()==%lu\n", "向推理空间拷贝", mat, q_mats.size());
////                checkRuntime(cudaMemcpyAsync(&gpuMemoryIn + count * inputSize, &mat, inputSize, cudaMemcpyDeviceToDevice, stream));
//            count += 1;
//            if (count >= 3) {
//                copy_over = true;
//                // 通知推理线程, 推理空间数据已经准备好
//                condition.notify_one();
//            }
//            printf("count = %d\n", count);
//            // 所有图片被抓取完且当队列为空,此时向该拷贝线程已不可能再工作了,退出
//            if (fetched_over && q_mats.empty()) {
//                break;
//            }
//        }
//
//    }
    while (true) {
        // 每次向推理空间gpuMemoryIn复制batchSize个处理好的图片
        {   // 消费时加锁
            std::unique_lock<std::mutex> l2(lock_);

//          当推理空间已保存图片数量为batchSize时, 等待, 等待推理线程把这批数据取走推理,然后再重新复制
            condition.wait(l2, [&]() {
                // false, 表示继续等待. true, 表示不等待,跳出wait
                // 当gpuMemoryIn填入预处理图片数据为batchSize时,执行推理. 或者最后图片数量不够一个batch也可以推理,此时q_mats一定是空的
                return !copy_over && !q_mats.empty();
            });
            auto mat = q_mats.front();
            //从前面删除
            q_mats.pop();
            // 消费掉一个,就可以通知队列跳出等待,继续生产
            condition.notify_one();
            printf("-------------------> %s mat: %d, q_mat.size()==%lu\n", "向推理空间拷贝", mat, q_mats.size());
//                checkRuntime(cudaMemcpyAsync(&gpuMemoryIn + count * inputSize, &mat, inputSize, cudaMemcpyDeviceToDevice, stream));
            this_thread::sleep_for(std::chrono::milliseconds(1));
            count += 1;

            printf(")))))))))))))) %d, %d\n", fetched_over, q_mats.empty());
//            // 满足一个batch,且刚好队列为空,队列也不在写入. 跳出循环,结束线程
//            if (count >= 3 && fetched_over && q_mats.empty()) {
//                transfer_finished = true;
//                break;
//            }
            // 队列为空,队列也不在写入, 不满足一个batch也直接跳出, 结束线程
            if (fetched_over && q_mats.empty()) {
                transfer_finished = true;
                //最后,若发现count>0,说明推理已经上传有图片,不论是否够一个batch,都视为拷贝完成,都要执行推理,
                if (count > 0) copy_over = true;
                condition.notify_all();
                break;
            }
            // 正常运行时,满足一个batch, 通知更新等待条件,通知推理线程, 推理空间数据已经准备好
            if (count >= 3) {
                copy_over = true;
                condition.notify_all();
            }
//            printf("count = %d\n", count);
        }
    }


    printf("a2 endl2\n");
}


void a3(int &count, bool &copy_flag, bool &transfer_finished) {
    while (true) {
        {
            std::unique_lock<std::mutex> l3(lock_);
            condition.wait(l3, [&]() {
                // false, 表示继续等待.
                // true, 表示不等待,跳出wait
                // 当gpuMemoryIn填入预处理图片数据为batchSize时,执行推理. 或者最后图片数量不够一个batch也可以推理,此时q_mats一定是空的
                return copy_flag;
            });
            // 执行异步推理
            printf("推理模拟-------------------->, count== %lu, q_mats.size() == %lu\n", count, q_mats.size());
        this_thread::sleep_for(std::chrono::milliseconds(1));

            copy_flag = false;
            count = 0;
            condition.notify_all();
            //流同步
        }

        // 小于3,说明不满足一个batch,已经取到最后了
        printf("trans = %d\n", transfer_finished);
        if (q_mats.empty() && transfer_finished) {
            //流同步后
            printf("### = %d\n", count);
            break;
        }

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
    for (int j = 0; j < 20000; ++j) {
        int count = 0;
        std::thread t1(a1, std::ref(fetch_over));
        std::thread t2(a2, std::ref(count), std::ref(fetch_over), std::ref(transfer_finish));
        std::thread t3(a3, std::ref(count), std::ref(copy_over), std::ref(transfer_finish));

        t1.join();
        t2.join();
        t3.join();
        fetch_over = false;
        copy_over = false;
        transfer_finish = false;
        printf("===========================================================================\n");
    }


    return 0;
}