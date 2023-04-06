//
// Created by Administrator on 2023/3/2.
//
#ifndef TENSORRTMODELDEPLOYMENT_INFER_CPP
#define TENSORRTMODELDEPLOYMENT_INFER_CPP
//编译用的头文件
#include <condition_variable>
#include "Infer.h"

#define checkRuntime(op) check_cuda_runtime((op),#op,__FILE__,__LINE__)
bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> ptrFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

// 推理输入
struct Job {
    float *inputTensor;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

// 推理输入图片路径+最终输出结果, 字段兼容各种输入类型. 不同输入类型对应传入对应字段中
struct futureJob {
    //取得是后处理后的结果
    std::shared_ptr<std::promise<batchBoxesType>> batchResult;

    std::string imgPath;
    std::vector<std::string> imgPaths;
    cv::Mat image;
    std::vector<cv::Mat> images;
    cv::cuda::GpuMat gpuImage;
    std::vector<cv::cuda::GpuMat> gpuImages;
};

// 推理输出
struct Out {
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
    static bool getEngineContext(ParamBase &curParam);
    //加载算法so文件
    static Infer *loadDynamicLibrary(const std::string &soPath);
    static std::vector<int> setBatchAndInferMemory(ParamBase &curParam);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    static unsigned long clcInputLength(const InputData &data);
//    std::shared_future<batchBoxesType> commit(const std::string &imagePath) override;
//    std::shared_future<batchBoxesType> commit(const cv::Mat &images) override;
//    std::shared_future<batchBoxesType> commit(const cv::cuda::GpuMat &images) override;
//
//    std::shared_future<batchBoxesType> commit(const std::vector<std::string> &imagePaths) override;
//    std::shared_future<batchBoxesType> commit(const std::vector<cv::Mat> &images) override;
//    std::shared_future<batchBoxesType> commit(const std::vector<cv::cuda::GpuMat> &images) override;
    std::shared_future<batchBoxesType> commit(const InputData &data) override;
    int preProcess(ParamBase &param, cv::Mat &image, float *pinMemoryCurrentIn) override {};

    int postProcess(ParamBase &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};

    void inferPre(ParamBase &curParam);
    void inferTrt(ParamBase &curParam);
    void inferPost(ParamBase &curParam, Infer *curFunc);

//    void inferPreBatch(ParamBase &curParam);
//    void inferTrtBatch(ParamBase &curParam);
//    void inferPostBatch(ParamBase &curParam, Infer *curFunc);

    bool startUpThread(ParamBase &curParam, Infer &curFunc);
//    bool startUpThreadBatch(ParamBase &param, Infer &curFunc);
private:
    std::mutex lock_;
    std::condition_variable cv_;

    float *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *pinMemoryIn = nullptr;
    float *gpuMemoryOut0 = nullptr, *gpuMemoryOut1 = nullptr, *pinMemoryOut = nullptr;
    float *gpuIn[2]{}, *gpuOut[2]{};
    std::vector<int> memory;
    // 读取从路径读入的图片矩阵
    cv::Mat mat;

    std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
    std::queue<Out> qOuts;
    futureJob fJob;
    std::queue<futureJob> qfJobs;

    // 记录传入的图片数量
    std::queue<int> qfJobLength;

    std::atomic<bool> preFinish{false};
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

#endif //TENSORRTMODELDEPLOYMENT_INFER_CPP