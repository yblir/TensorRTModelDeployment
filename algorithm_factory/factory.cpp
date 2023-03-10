//
// Created by 12134 on 2023/2/9.
//

#include "factory.h"

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

void TRTLogger::log(nvinfer1::ILogger::Severity severity, const nvinfer1::AsciiChar *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        // 打印带颜色的字符，格式如下：
        // printf("\033[47;33m打印的文本\033[0m");
        // 其中 \033[ 是起始标记
        //      47    是背景颜色
        //      ;     分隔符
        //      33    文字颜色
        //      m     开始标记结束
        //      \033[0m 是终止标记
        // 其中背景颜色或者文字颜色可不写
        // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
        if (severity == Severity::kWARNING) {
            printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
        } else if (severity <= Severity::kERROR) {
            printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
        } else {
            printf("%s: %s\n", severity_string(severity), msg);
        }
    }
}

//构建引擎
bool AlgorithmBase::buildEngine(const std::string &onnxFilePath, const std::string &saveEnginePath, int maxBatch) {
    //检查待转换的onnx文件是否存在
    if (!std::filesystem::exists(onnxFilePath)) {
        std::cout << "path not exist: " << onnxFilePath << std::endl;
        return false;
    }

    TRTLogger logger;
    uint32_t flag = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //1 基本组件, 不想麻烦就就全部写成auto类型~
    auto builder = prtFree(nvinfer1::createInferBuilder(logger));
    auto config = prtFree(builder->createBuilderConfig());
    std::shared_ptr<nvinfer1::INetworkDefinition> network = prtFree(builder->createNetworkV2(flag));

    //2 通过onnxparser解析器的结果,填充到network中,类似addconv的方式添加进去
    auto parser = prtFree(nvonnxparser::createParser(*network, logger));
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
    std::shared_ptr<nvinfer1::IHostMemory> serializedModel = prtFree(builder->buildSerializedNetwork(*network, *config));
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


AlgorithmBase *AlgorithmBase::loadDynamicLibrary(const std::string &soPath) {
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

    if (nullptr == clct) {
        return nullptr;
    }

    return clct();
}

//从硬盘加载引擎文件到内存
std::vector<unsigned char> AlgorithmBase::loadEngine(const std::string &engineFilePath) {
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

std::string AlgorithmBase::getEnginePath(const ParamBase &conf) {
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

std::shared_ptr<nvinfer1::ICudaEngine> AlgorithmBase::createEngine(const std::vector<unsigned char> &engineFile) {
    TRTLogger logger;

    auto runtime = prtFree(nvinfer1::createInferRuntime(logger));
    auto engine = prtFree(runtime->deserializeCudaEngine(engineFile.data(), engineFile.size()));

    if (nullptr == engine) {
        printf("deserialize cuda engine failed.\n");
        return nullptr;
    }
//     engine->getNbIOTensors()-1,是因为有1个是输入.剩下的才是输出
    if (2 != engine->getNbIOTensors()) {
        printf("create engine failed: onnx导出有问题,必须是1个输入和1个输出,当前有%d个输出\n", engine->getNbIOTensors() - 1);
        return nullptr;
    }

    return engine;
}


