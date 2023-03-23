//
// Created by Administrator on 2023/3/2.
//

#include "Infer.h"

class InferImpl : public Infer {
public:
    std::shared_ptr<std::string> commit(const std::string& input) override;
    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParmBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);
    // 创建推理engine
    static std::shared_ptr<nvinfer1::ICudaEngine> createEngine(const std::vector<unsigned char> &engineFile);
    //加载算法so文件
    static Infer *loadDynamicLibrary(const std::string &soPath);
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