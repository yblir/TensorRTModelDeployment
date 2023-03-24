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
#include "Infer.h"

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

// 推理输出
struct Out {
    std::shared_future<float *> inferResult;
    std::vector<std::vector<float>> d2is;
    int inferNum{};
};

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
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
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
    InferImpl();
    // 获得引擎名字, conf: 对应具体实现算法的结构体引用
    static std::string getEnginePath(const ParamBase &conf);
    //构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    static bool buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch);
    //加载引擎到gpu,准备推理.
    static std::vector<unsigned char> loadEngine(const std::string &engineFilePath);

//    std::shared_ptr<std::string> commit(const std::string &input) override;

    // 创建推理engine
    static bool createInfer(ParamBase &param, const std::string &enginePath);
    //加载算法so文件
    static Infer *loadDynamicLibrary(const std::string &soPath);

//    bool startThread();

    std::vector<std::shared_future<std::string>> commit(const std::vector<std::string>& imagePaths) override;
private:
    //struct
    std::mutex lock_;
    std::condition_variable cv_;

    std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
//    std::queue<Out> qOuts;
    std::queue<float *> qOuts;

    std::atomic<bool> queueFinish{false};
    std::atomic<bool> inferFinish{false};

    std::thread preThread;
    std::thread inferThread;
    std::thread postThread;

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
    std::shared_ptr<nvinfer1::IHostMemory> serializedModel = ptrFree(builder->buildSerializedNetwork(*network, *config));
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

    return true;
}

std::vector<std::shared_future<std::string>> InferImpl::commit(const std::vector<std::string> &imagePaths) {

    return 0;
}

InferImpl::InferImpl() {
    cudaStreamCreate(&preStream);
    cudaStreamCreate(&inferStream);
    cudaStreamCreate(&postStream);
}

std::shared_ptr<Infer> createInfer(ParamBase &param, const std::string &enginePath) {
    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl());
    // 如果创建引擎不成功就reset
    if (!instance->createInfer(param, enginePath)) instance.reset();
    if (instance) instance.startThread
    return instance;
}

