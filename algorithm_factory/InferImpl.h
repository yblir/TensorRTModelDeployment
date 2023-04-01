//
// Created by Administrator on 2023/3/2.
//
#ifndef TENSORRTMODELDEPLOYMENT_INFER_CPP
#define TENSORRTMODELDEPLOYMENT_INFER_CPP
//编译用的头文件
//#include <NvInfer.h>
#include <condition_variable>
#include "Infer.h"

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> ptrFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

// 推理输入
struct Job {
//    std::shared_ptr<std::promise<float *>> gpuOutputPtr;
//    std::shared_ptr<float> inputTensor{};
    float *inputTensor;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

// 推理输入图片路径+最终输出结果
struct futureJob {
    //取得是后处理后的结果
    std::shared_ptr<std::promise<batchBoxesType>> batchResult;
//    float *inputTensor{};
    std::vector<std::string> imgPaths;
};

//// 推理输出
//struct Out {
//    std::shared_future<float *> inferResult;
//    std::vector<std::vector<float>> d2is;
//    int inferNum{};
//};

// 推理输出
struct Out {
//    std::shared_ptr<float> inferOut;
    float *inferOut;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

class InferImpl : public Infer {
public:
    explicit InferImpl(std::vector<int> &memory);
    ~InferImpl() override;

    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParamBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);

    // 创建推理engine
    static bool getEngineContext(ParamBase &param, const std::string &enginePath);
    //加载算法so文件
    static Infer *loadDynamicLibrary(const std::string &soPath);

    bool startUpThread(ParamBase &param, Infer &curFunc);
    std::shared_future<batchBoxesType> commit(const std::vector<std::string> &imagePaths) override;
    static std::vector<int> setBatchAndInferMemory(ParamBase &curParam);

    int preProcess(ParamBase &param, cv::Mat &image, float *pinMemoryCurrentIn) override {};

    int postProcess(ParamBase &param, float *pinMemoryOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};

    void inferPre(ParamBase &curParam);
    void inferTrt(ParamBase &curParam);
    void inferPost(ParamBase &curParam, Infer *curFunc);

private:
    std::mutex lock_;
    std::condition_variable cv_;
//    std::shared_ptr<float *> *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *pinMemoryIn = nullptr;
//    std::shared_ptr<float *> *gpuMemoryOut0 = nullptr, *gpuMemoryOut1 = nullptr, *pinMemoryOut = nullptr;
//    std::shared_ptr<float *> gpuIn[2], gpuOut[2];
    float *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *pinMemoryIn = nullptr;
    float *gpuMemoryOut0 = nullptr, *gpuMemoryOut1 = nullptr, *pinMemoryOut = nullptr;
    float *gpuIn[2]{}, *gpuOut[2]{};
    std::vector<int> memory;
    // 读取从路径读入的图片矩阵
    cv::Mat mat;

    std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
    std::queue<Out> qOuts;
    futureJob fJob1;
    std::queue<futureJob> qJob2;

    // 记录传入的图片数量
    std::queue<int> qfJobLength;

    std::queue<std::vector<std::string >> qPaths;

    std::atomic<bool> queueFinish{false};
    std::atomic<bool> inferFinish{false};
    std::atomic<bool> workRunning{true};

    std::shared_ptr<std::thread> preThread;
    std::shared_ptr<std::thread> inferThread;
    std::shared_ptr<std::thread> postThread;

    //创建cuda任务流,对应上述三个处理线程
    cudaStream_t preStream{};
    cudaStream_t inferStream{};
    cudaStream_t postStream{};
};

//std::shared_ptr<Infer> createInfer(ParamBase &param, const std::string &enginePath, Infer &curFunc) {
//    std::vector<int> memory = InferImpl::setBatchAndInferMemory(param);
//
//    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
//    std::shared_ptr<InferImpl> instance(new InferImpl(memory));
//
//    // 如果创建引擎不成功就reset
//    if (!instance->getEngineContext(param, enginePath)) {
//        instance.reset();
//        return instance;
//    }
//
//    // 若线程启动失败,也返回空实例. 所有的错误信息都在函数内部打印
//    if (!instance->startUpThread(param, curFunc)) {
//        instance.reset();
//        return instance;
//    }
//    return instance;
//}

#endif //TENSORRTMODELDEPLOYMENT_INFER_CPP