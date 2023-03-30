//
// Created by Administrator on 2023/3/2.
//
// 所有待加速部署的模型,都继承工厂类,依据实际需求各自实现自己的代码
#include <iostream>
#include <istream>
#include <cuda_runtime.h>
//编译用的头文件
#include <NvInfer.h>
//onnx解释器头文件
#include <NvOnnxParser.h>
//推理用的运行时头文件
#include <NvInferRuntime.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <dlfcn.h>
#include <opencv2/opencv.hpp>
#include <utility>
#include "Infer.h"

using batchBoxesType = std::vector<std::vector<std::vector<float>>>;

//通过智能指针管理nv, 内存自动释放,避免泄露.
template<typename T>
std::shared_ptr<T> ptrFree(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *p) { delete p; });
}

// 推理输入
struct Job {
    std::shared_ptr<std::promise<float *>> gpuOutputPtr;
    float *inputTensor{};
};

// 推理输入
struct Job2 {
    std::shared_ptr<std::promise<std::map<std::string, batchBoxesType>>> gpuOutputPtr;
//    float *inputTensor{};
    std::vector<std::string> paths;
};

// 推理输出
struct Out {
    std::shared_future<float *> inferResult;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

#define checkRuntime(op) check_cuda_runtime((op),#op,__FILE__,__LINE__)

bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line) {
    if (cudaSuccess != code) {
        const char *errName = cudaGetErrorName(code);
        const char *errMsg = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, errName, errMsg);
        return false;
    }
    return true;
}

const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            } else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            } else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    };
};


class InferImpl : public Infer {
public:
    InferImpl(std::vector<int> &memory);
    ~InferImpl() override;

    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParamBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);

    // 创建推理engine
    static bool createInfer(ParamBase &param, const std::string &enginePath);
    //加载算法so文件
    static Infer *loadDynamicLibrary(const std::string &soPath);

    bool startThread(ParamBase &param, std::vector<int> &memory);
    std::shared_future<std::map<std::string, batchBoxesType>> commit(const std::vector<std::string> &imagePaths) override;
    static std::vector<int> setBatchAndInferMemory(ParamBase &curParam);

//    std::vector<int> getMemory
    int preProcess(ParamBase &param, cv::Mat &image, float *pinMemoryCurrentIn) override {};

    int postProcess(ParamBase &param, float *pinMemoryOut, int singleOutputSize,
                    int outputNums, batchBoxesType &result) override {};

    void inferPre(ParamBase &curParam, std::vector<int> &memory);
    int inferTrt(ParamBase &curParam, std::vector<int> &memory);
    void inferPost(ParamBase &curParam, AlgorithmBase *curFunc, std::vector<int> &memory,
                   std::shared_future<batchBoxesType> &result);
    void getMemory(std::vector<int> memory_);
private:
    std::mutex lock_;
    std::condition_variable cv_;
    float *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *pinMemoryIn = nullptr;
    float *gpuMemoryOut0 = nullptr, *gpuMemoryOut1 = nullptr, *pinMemoryOut = nullptr;
    float *gpuIn[2], *gpuOut[2];
    std::vector<int> memory;

    Job preJob;
    Job trtJob;
    Out preOut;
    Out postOut;
    Job2 job2;
//    std::vector<int> memory = InferImpl::setBatchAndInferMemory(curParam);
    std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
//    std::queue<Out> qOuts;
    std::queue<float *> qOuts;

    std::queue<Job2> qJob2;

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

std::string InferImpl::getEnginePath(const ParamBase &conf) {
    int num;
    // 检查指定编号的显卡是否正常
    cudaError_t cudaStatus = cudaGetDeviceCount(&num);
    if ((cudaSuccess != cudaStatus) || (num == 0) || (conf.gpuId > (num - 1))) {
        printf("infer device id: %d error or no this gpu.\n", conf.gpuId);
        return "";
    }

    cudaDeviceProp prop{};
    std::string gpuName, enginePath;
    // 获取显卡信息
    cudaStatus = cudaGetDeviceProperties(&prop, conf.gpuId);

    if (cudaSuccess == cudaStatus) {
        gpuName = prop.name;
        // 删掉显卡名称内的空格
        gpuName.erase(std::remove_if(gpuName.begin(), gpuName.end(), ::isspace), gpuName.end());
    } else {
        printf("get device info error.\n");
        return "";
    }

    std::string strFp16 = conf.useFp16 ? "Fp16" : "Fp32";

    // 拼接待构建或加载的引擎路径
    enginePath = conf.onnxPath.substr(0, conf.onnxPath.find_last_of('.')) + "_" + gpuName + "_" + strFp16 + ".engine";

    return enginePath;
}

//构建引擎
bool InferImpl::buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch) {
    //检查待转换的onnx文件是否存在
    if (!std::filesystem::exists(onnxFilePath)) {
        std::cout << "path not exist: " << onnxFilePath << std::endl;
        return false;
    }

    TRTLogger logger;
    uint32_t flag = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //1 基本组件, 不想麻烦就就全部写成auto类型~
    auto builder = ptrFree(nvinfer1::createInferBuilder(logger));
    auto config = ptrFree(builder->createBuilderConfig());
    std::shared_ptr<nvinfer1::INetworkDefinition> network = ptrFree(builder->createNetworkV2(flag));

    //2 通过onnxparser解析器的结果,填充到network中,类似addconv的方式添加进去
    auto parser = ptrFree(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnxFilePath.c_str(), 1)) {
        printf("failed to parse onnx file\n");
        return false;
    }

    //3 设置最大工作缓存
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 28);


    //4 若模型有多个输入,必须使用多个profile,好在一般情况想下都是1
    auto profile = builder->createOptimizationProfile();

    auto inputTensor = network->getInput(0);  //取得第一个输入,一般输入batch都是1,0号索引必定能取得,1号索引可以不?
    auto inputDims = inputTensor->getDimensions();

    //配置最小最大范围, 目标检测领域,输入尺寸是不变的,变化的batch
    inputDims.d[0] = 1;
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, inputDims);
    inputDims.d[0] = maxBatch;
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

    //5 直接序列化推理引擎
    std::shared_ptr<nvinfer1::IHostMemory> serializedModel = ptrFree(
            builder->buildSerializedNetwork(*network, *config));
    if (nullptr == serializedModel) {
        printf("build engine failed\n");
        return false;
    }

    //6 将推理引擎序列化,存储为文件
    auto *f = fopen(saveEnginePath.c_str(), "wb");
    fwrite(serializedModel->data(), 1, serializedModel->size(), f);
    fclose(f);

    printf("build and save engine success \n");

    return true;
}

//从硬盘加载引擎文件到内存
std::vector<unsigned char> InferImpl::loadEngine(const std::string &engineFilePath) {
    //检查engine文件是否存在
    if (!std::filesystem::exists(engineFilePath)) {
        std::cout << "path not exist: " << engineFilePath << std::endl;
        return {};
    }

    std::ifstream in(engineFilePath, std::ios::in | std::ios::binary);
    if (!in.is_open()) { return {}; }

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *) &data[0], length);
    }
    in.close();

    return data;
}

Infer *InferImpl::loadDynamicLibrary(const std::string &soPath) {
    // todo 这里,要判断soPath是不是合法. 并且是文件名时搜索路径
    printf("load dynamic lib: %s\n", soPath.c_str());
    auto soHandle = dlopen(soPath.c_str(), RTLD_NOW);
    if (!soHandle) {
        printf("open dll error: %s \n", dlerror());
        return nullptr;
    }
    //找到符号Make_Collector的地址
    void *void_ptr = dlsym(soHandle, "MakeAlgorithm");
    char *error = dlerror();
    if (nullptr != error) {
        printf("dlsym error:%s\n", error);
        return nullptr;
    }
    auto tmp = reinterpret_cast<ptrdiff_t> (void_ptr);
    auto clct = reinterpret_cast< AlgorithmCreate >(tmp);

    if (nullptr == clct) return nullptr;

    return clct();
}

//使用所有加速算法的初始化部分: 初始化参数,构建engine, 反序列化制作engine
bool InferImpl::createInfer(ParamBase &param, const std::string &enginePath) {
    //获取engine绝对路径
    param.enginePath = InferImpl::getEnginePath(param);

    std::vector<unsigned char> engineFile;
    // 判断引擎文件是否存在,如果不存在,要先构建engine
    if (std::filesystem::exists(param.enginePath))
        // engine存在,直接加载engine文件,反序列化引擎文件到内存
        engineFile = InferImpl::loadEngine(param.enginePath);
    else {
        //engine不存在,先build,序列化engine到硬盘, 再执行反序列化操作
        if (InferImpl::buildEngine(param.onnxPath, param.enginePath, param.batchSize))
            engineFile = InferImpl::loadEngine(param.enginePath);
    }

    if (engineFile.empty()) return false;

//   也可以直接从字符串提取名字 param.enginePath.substr(param.enginePath.find_last_of("/"),-1)
    std::cout << "start create engine: " << std::filesystem::path(param.enginePath).filename() << std::endl;

    // 创建engine并获得执行上下文context =======================================================================================
    TRTLogger logger;
    auto runtime = ptrFree(nvinfer1::createInferRuntime(logger));
    param.engine = ptrFree(runtime->deserializeCudaEngine(engineFile.data(), engineFile.size()));

    if (nullptr == param.engine) {
        printf("deserialize cuda engine failed.\n");
        return false;
    }
//    是因为有1个是输入.剩下的才是输出
    if (2 != param.engine->getNbIOTensors()) {
        printf("create engine failed: onnx导出有问题,必须是1个输入和1个输出,当前有%d个输出\n", param.engine->getNbIOTensors() - 1);
        return false;
    }
    if (nullptr == param.engine) {
        std::cout << "failed engine name: " << std::filesystem::path(param.enginePath).filename() << std::endl;
        return false;
    }
    param.context = ptrFree(param.engine->createExecutionContext());
    std::cout << "create engine and context success: " << std::filesystem::path(param.enginePath).filename() << std::endl;
//    worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
    return true;
}

std::shared_future<std::map<std::string, batchBoxesType>> InferImpl::commit(const std::vector<std::string> &imagePaths) {
    // 将传入的多张或一张图片,一次性传入队列总
    Job2 job3;
    job3.paths = imagePaths;
    job3.gpuOutputPtr.reset(new std::promise<std::map<std::string, batchBoxesType>>());
    // 创建share_future变量,一次性取回传入的所有图片结果
    std::shared_future<std::map<std::string, batchBoxesType>> fut = job3.gpuOutputPtr->get_future();
    {
        std::lock_guard<std::mutex> l(lock_);
        //
        qJob2.emplace(std::move(job3));
    }
    // 通知图片预处理线程,图片以及就位.
    cv_.notify_one();
    return fut;
}

void InferImpl::inferPre(ParamBase &curParam, std::vector<int> &memory) {
    // 记录预处理总耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();

    unsigned long mallocSize = memory[0] * sizeof(float);

    Job2 job_temp;
    {
        std::unique_lock<std::mutex> l(lock_);
        cv_.wait(l, [&]() { return !qJob2.empty() || !workRunning; });
        if (!workRunning) return;
        job_temp = qJob2.front();
        qJob2.pop();
    }
    auto lastAddress = &job_temp.paths.back();

    // count统计推理图片数量,正常情况下等于batchSize,但在最后一次,可能小于batchSize.imgNums统计预处理图片数量,index是gpuIn的索引,三个显存轮换使用
    int count = 0, imgNums = 0, index = 0, inputSize = memory[0] / curParam.batchSize;
    Job job;
    Out out;
    cv::Mat mat;

    for (auto &imgPath: job_temp.paths) {
        // auto stream = cv::cuda::StreamAccessor::wrapStream(m_stream);
        mat = cv::imread(imgPath);
        // 记录前处理耗时
        auto preTime = timer.curTimePoint();
        float d2i[6];
        cv::Mat scaleImage = letterBox(mat, curParam.inputWidth, curParam.inputHeight, d2i);
//      -------------------------------------------------------------------------
//        cv::cuda::GpuMat gpuMat(scaleImage);
//        cv::cuda::cvtColor(gpuMat, gpuMat, cv::COLOR_BGR2RGB);
//        // 数值归一化
//        gpuMat.convertTo(gpuMat, CV_32FC3, 1. / 255., 0);
//      -------------------------------------------------------------------------
        // 依次存储一个batchSize中图片放射变换参数
        out.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
        if (count < curParam.batchSize) {
            BGR2RGB(scaleImage, pinMemoryIn + count * inputSize);
//            checkRuntime(cudaMemcpy(gpuMemoryIn + count * inputSize, &gpuMat, memory[0] * sizeof(float), cudaMemcpyDeviceToDevice));
            count += 1;
        }

        // 小于一个batch,不再往下继续.
        if (count < curParam.batchSize) continue;
        // 统计图片尺寸变换等预处理耗时
        totalTime += timer.timeCount(preTime);

        //全部到gpu上,不需要这句复制
        checkRuntime(cudaMemcpyAsync(gpuIn[index], pinMemoryIn, mallocSize, cudaMemcpyHostToDevice, stream));
//            checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn, memory[0] * sizeof(float), cudaMemcpyHostToDevice, stream));
        job.inputTensor = gpuIn[index];
        // 记录当前推理图片数量,正常情况count=batchSize, 但最后批次,可能会小于batchSize
//        out.inferNum = count;
        // 以后取回该批次图片的结果
//        job.gpuOutputPtr.reset(new std::promise<float *>());
        // 关联当前输入数据的未来输出
//        out.inferResult = job.gpuOutputPtr->get_future();

        // 流同步,在写入队列前,确保待推理数据已复制到gpu上
        checkRuntime(cudaStreamSynchronize(stream));

        // 加锁
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait. wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
            cv_.wait(l, [&]() { return qJobs.size() < 3; });

            // 将存有推理数据的显存空间指针保存
            qJobs.push(job);
//            qOuts.push(out);
            cv_.notify_one();
        }

        imgNums += count;
        count = 0;
        // 轮换到下一个显存空间
        index = index >= 1 ? 0 : index + 1;
        // 清空保存仿射变换参数,每个out只保留一个batch的参数
        if (&imgPath != lastAddress) std::vector<std::vector<float>>().swap(out.d2is);

    }
    // 所有图片处理完,处理不满足一个batchSize的情况
    if (count != 0 && count < curParam.batchSize) {
        checkRuntime(cudaMemcpy(gpuIn[index], pinMemoryIn, count * inputSize * sizeof(float), cudaMemcpyHostToDevice));
        job.inputTensor = gpuIn[index];
//        out.inferNum = count;
        imgNums += count;
        // 以后取回该批次图片的结果
//        job.gpuOutputPtr.reset(new std::promise<float *>());
        // 关联当前输入数据的未来输出
//        out.inferResult = job.gpuOutputPtr->get_future();
        // 加锁
        {
            std::unique_lock<std::mutex> l(lock_);
            cv_.wait(l, [&]() { return qJobs.size() < 3; });
            // 将存有推理数据的显存空间指针保存
            qJobs.push(job);
//            qOuts.push(out);
            cv_.notify_one();
        }
    }
    queueFinish = true;

    // 线程结束时,仅是否pinMemoryIn, gpuMemoryIn存有推理数据,在推理线程中释放
//    for (auto &item: gpuIn) checkRuntime(cudaFreeAsync(item, stream));
//    checkRuntime(cudaFreeHost(pinMemoryIn));
    totalTime2 = timer.timeCount(t);
    printf("pre   use time: %.2f ms, thread use time: %.2f ms, pre img num = %d\n", totalTime, totalTime2, imgNums);
    checkRuntime(cudaStreamDestroy(stream));
}

// 适用于模型推理的通用trt流程
int InferImpl::inferTrt(ParamBase &curParam, std::vector<int> &memory) {
    // 记录推理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
    unsigned long mallocSize = memory[1] * sizeof(float);
    //创建cuda任务流
//    cudaStream_t stream;
//    checkRuntime(cudaStreamCreate(&stream));

//    float *gpuMemoryOut = nullptr;
//    checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
    curParam.context->setTensorAddress(curParam.outputName.c_str(), gpuMemoryOut);
    Job job;
    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            cv_.wait(l, [&]() { return !qJobs.empty(); });
            job = qJobs.front();
            qJobs.pop();
            // 消费掉一个元素,通知队列跳出等待,继续生产
            cv_.notify_one();
        }
        curParam.context->setTensorAddress(curParam.inputName.c_str(), job.inputTensor);

        auto qT1 = timer.curTimePoint();
        curParam.context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        totalTime += timer.timeCount(qT1);
        // 流同步后,获取该batchSize推理结果
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait. wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
            cv_.wait(l, [&]() { return qOuts2.size() < 2; });
            qOuts2.push(gpuMemoryOut);
            cv_.notify_one();
        }

        // 再判断一次推理数据队列,若图片都处理完且队列空,说明推理结束,不必再开辟新空间了,直接退出线程
        if (queueFinish && qJobs.empty()) break;

        // 重新开辟推理输出空间,保证每个推理输出空间都不同,避免多线程推理时的结果覆盖
//        checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
    }
    inferFinish = true;
    totalTime2 = timer.timeCount(t);
    printf("infer use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
    checkRuntime(cudaStreamDestroy(stream));
}

void
InferImpl::inferPost(ParamBase &curParam, AlgorithmBase *curFunc, std::vector<int> &memory, std::shared_future<batchBoxesType> &result) {
    // 记录后处理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
//    std::shared_future<std::vector<std::vector<std::vector<float>>>> result;

    int mallocSize = memory[1] * sizeof(float), singleOutputSize = memory[1] / curParam.batchSize;
//    float *pinMemoryOut = nullptr;
    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, mallocSize));

    float *out;
    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            cv_.wait(l, [&]() { return !qOut2.empty() || (inferFinish && qOut2.empty()); });
            // 退出推理线程
            if (inferFinish && qOuts2.empty()) return;

            out = qOuts2.front();
            // 把当前取出的元素删掉
            qOuts2.pop();
        }
    }
//        float *futureResult = out.inferResult.get();
    // 转移到内存中处理
    cudaMemcpy(pinMemoryOut, out, mallocSize, cudaMemcpyDeviceToHost);
    curParam.d2is = out.d2is;
    auto qT1 = timer.curTimePoint();
    curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, out.inferNum, result);
    totalTime += timer.timeCount(qT1);

    // 清理已用完的gpuMemoryOut
//        checkRuntime(cudaFreeAsync(futureResult, stream));


    // 后处理全部完成,释放存储推理结果的pinMemoryOut
//    if (pinMemoryOut) checkRuntime(cudaFreeHost(pinMemoryOut));
    totalTime2 = timer.timeCount(t);
    printf("post  use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);

//    return result;
}

std::vector<int> InferImpl::setBatchAndInferMemory(ParamBase &curParam) {
    //计算输入tensor所占存储空间大小,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = curParam.engine->getTensorShape(curParam.inputName.c_str());
    inputShape.d[0] = curParam.batchSize;
    curParam.context->setInputShape(curParam.inputName.c_str(), inputShape);
    //batchSize * c * h * w
    int inputSize = curParam.batchSize * inputShape.d[1] * inputShape.d[2] * inputShape.d[3];

    // 获得输出tensor形状,计算输出所占存储空间
    auto outputShape = curParam.engine->getTensorShape(curParam.outputName.c_str());
    // 记录这两个输出维度数量,在后处理时会用到
    curParam.predictNums = outputShape.d[1];
    curParam.predictLength = outputShape.d[2];
    // 计算推理结果所占内存空间大小
    int outputSize = curParam.batchSize * outputShape.d[1] * outputShape.d[2];

    // 使用元组返回多个多个不同的类型值, 供函数外调用,有以下两种方式可使用
//    return std::make_tuple(inputShape, inputSize, outputSize);
//    return std::tuple<nvinfer1::Dims32, int, int>{inputShape, inputSize, outputSize};

    // 将输入输出所占空间大小返回
    std::vector<int> memory = {inputSize, outputSize};
    return memory;
}

bool InferImpl::startThread(ParamBase &param, std::vector<int> &memory) {
    preThread = std::make_shared<std::thread>(&InferImpl::inferPre, this, std::ref(param), std::ref(memory));
    inferThread = std::make_shared<std::thread>(&InferImpl::inferTrt, this, std::ref(param), std::ref(memory));
    postThread = std::make_shared<std::thread>(&InferImpl::inferPost, this, std::ref(param),
                                               std::ref(curFunc), std::ref(memory), std::ref(result));
}

void InferImpl::getMemory(std::vector<int> memory_) {
    this->memory = std::move(memory_);
}

InferImpl::InferImpl(std::vector<int> &memory) {
//    this->memory = memory;

    cudaStream_t initStream;
    cudaStreamCreate(&initStream);

    cudaStreamCreate(&preStream);
    cudaStreamCreate(&inferStream);
    cudaStreamCreate(&postStream);
//    getMemory(memory);
    unsigned long InMallocSize = memory[0] * sizeof(float);
    unsigned long OutMallocSize = memory[1] * sizeof(float);

    checkRuntime(cudaMallocAsync(&gpuMemoryIn0, InMallocSize, initStream));
    checkRuntime(cudaMallocAsync(&gpuMemoryIn1, InMallocSize, initStream));

    checkRuntime(cudaMallocAsync(&gpuMemoryOut0, OutMallocSize, initStream));
    checkRuntime(cudaMallocAsync(&gpuMemoryOut1, OutMallocSize, initStream));

    checkRuntime(cudaMallocHost(&pinMemoryOut, InMallocSize));
    checkRuntime(cudaMallocHost(&pinMemoryIn, OutMallocSize));

    gpuIn[0] = gpuMemoryIn0;
    gpuIn[1] = gpuMemoryIn1;
    gpuOut[0] = gpuMemoryOut0;
    gpuOut[1] = gpuMemoryOut1;

    checkRuntime(cudaStreamDestroy(initStream));
}

InferImpl::~InferImpl() {
//    printf('d');
}


std::shared_ptr<Infer> createInfer(ParamBase &param, const std::string &enginePath) {
    std::vector<int> memory = InferImpl::setBatchAndInferMemory(param);
    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl(memory));
    // 如果创建引擎不成功就reset
    if (!instance->createInfer(param, enginePath)) {
        instance.reset();
        return instance;
    }
    if (instance) {
        instance->startThread(param, memory);
    }
    return instance;
}

