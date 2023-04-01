//
// Created by Administrator on 2023/4/1.
//
#include <cuda_runtime.h>
//onnx解释器头文件
#include <NvOnnxParser.h>
#include <fstream>
#include <dlfcn.h>
#include "../utils/general.h"
#include "../utils/box_utils.h"

#include "InferImpl.h"

Timer timer = Timer();

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
    }
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
bool InferImpl::getEngineContext(ParamBase &param, const std::string &enginePath) {
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

std::shared_future<batchBoxesType> InferImpl::commit(const std::vector<std::string> &imagePaths) {
    // 将传入的多张或一张图片,一次性传入队列总
//    futureJob job;
    fJob1.imgPaths = imagePaths;
    fJob1.batchResult.reset(new std::promise<batchBoxesType>());
    // 创建share_future变量,一次性取回传入的所有图片结果
    std::shared_future<batchBoxesType> fut = fJob1.batchResult->get_future();
    {
        std::lock_guard<std::mutex> l(lock_);
        qfJobLength.emplace(fJob1.imgPaths.size());
        qJob2.emplace(std::move(fJob1));
    }
    // 通知图片预处理线程,图片已就位.
    cv_.notify_one();

    return fut;
}

void InferImpl::inferPre(ParamBase &curParam) {
    // 记录预处理总耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
    unsigned long InMallocSize = memory[0] * sizeof(float);
    checkRuntime(cudaMalloc(&gpuMemoryIn0, InMallocSize));
    checkRuntime(cudaMalloc(&gpuMemoryIn1, InMallocSize));
    gpuIn[0] = gpuMemoryIn0;
    gpuIn[1] = gpuMemoryIn1;

    unsigned long mallocSize = this->memory[0] * sizeof(float);
    // count统计推理图片数量,最后一次,可能小于batchSize.imgNums统计预处理图片数量,index是gpuIn的索引,两个显存轮换使用
    int count = 0, index = 0, inputSize = this->memory[0] / curParam.batchSize;
    float d2i[6];
    Job job;

    while (workRunning) {
        {
            std::unique_lock<std::mutex> l(lock_);
            cv_.wait(l, [&]() { return !qJob2.empty() || !workRunning; });
            if (!workRunning) break;
            fJob1 = qJob2.front();
            qJob2.pop();
            printf("fjob1 pop\n");
        }
        // 默认推理的数量为batchSize, 只有最后一次才可能小于batchSize
        job.inferNum = curParam.batchSize;

        for (auto &imgPath: fJob1.imgPaths) {
            mat = cv::imread(imgPath);
            // 记录前处理耗时
            auto preTime = timer.curTimePoint();
            cv::Mat scaleImage = letterBox(mat, curParam.inputWidth, curParam.inputHeight, d2i);

            // 依次存储一个batchSize中图片放射变换参数
            job.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
            if (count < curParam.batchSize) {
                BGR2RGB(scaleImage, pinMemoryIn + count * inputSize);
                count += 1;
            }
            // 小于一个batch,不再往下继续.
            if (count < curParam.batchSize) continue;
            totalTime += timer.timeCount(preTime);

            //全部到gpu上,不需要这句复制
//            checkRuntime(cudaMemcpy(gpuIn[index], pinMemoryIn, mallocSize, cudaMemcpyHostToDevice));
            checkRuntime(cudaMemcpyAsync(gpuIn[index], pinMemoryIn, mallocSize, cudaMemcpyHostToDevice, preStream));
            job.inputTensor = gpuIn[index];
            // 流同步,在写入队列前,确保待推理数据已复制到gpu上
            checkRuntime(cudaStreamSynchronize(preStream));

            {
                std::unique_lock<std::mutex> l(lock_);
                // false,继续等待. true,不等待,跳出wait. 一旦进入wait,解锁. 退出wait,又加锁 最多2个batch
                cv_.wait(l, [&]() { return qJobs.size() < 2; });
                // 将一个batch待推理数据的显存空间指针保存
                qJobs.push(job);
                cv_.notify_one();
                printf("qjobs pushh\n");
            }


            count = 0;
            // 轮换到下一个显存空间索引号
            index = index >= 1 ? 0 : index + 1;
            // 清空保存仿射变换参数,只保留当前推理batch的参数
            std::vector<std::vector<float>>().swap(job.d2is);
        }
        // 所有图片处理完,处理不满足一个batchSize的情况
        if (count != 0 && count < curParam.batchSize) {
            checkRuntime(cudaMemcpy(gpuIn[index], pinMemoryIn, count * inputSize * sizeof(float), cudaMemcpyHostToDevice));
            job.inputTensor = gpuIn[index];
            job.inferNum = count;

            {
                std::unique_lock<std::mutex> l(lock_);
                cv_.wait(l, [&]() { return qJobs.size() < 2; });
                qJobs.push(job);
                cv_.notify_one();
            }

        }
    }
    queueFinish = true;
    totalTime2 = timer.timeCount(t);
    printf("pre   use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
}

// 适用于模型推理的通用trt流程
void InferImpl::inferTrt(ParamBase &curParam) {
    // 记录推理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
    int index2 = 0;

    unsigned long OutMallocSize = memory[1] * sizeof(float);

    checkRuntime(cudaMalloc(&gpuMemoryOut0, OutMallocSize));
    checkRuntime(cudaMalloc(&gpuMemoryOut1, OutMallocSize));
    gpuOut[0] = gpuMemoryOut0;
    gpuOut[1] = gpuMemoryOut1;

    Job job;
    Out out;

    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            cv_.wait(l, [&]() { return !qJobs.empty(); });
            job = qJobs.front();
            qJobs.pop();
            printf("trttttt");
            // 消费掉一个元素,通知队列跳出等待,继续生产
            cv_.notify_one();
        }

        auto qT1 = timer.curTimePoint();

        curParam.context->setTensorAddress(curParam.inputName.c_str(), &job.inputTensor);
        curParam.context->setTensorAddress(curParam.outputName.c_str(), &gpuOut[index2]);
        curParam.context->enqueueV3(inferStream);
        cudaStreamSynchronize(inferStream);
        printf("单次推理结束\n");
        totalTime += timer.timeCount(qT1);
        index2 = index2 >= 1 ? 0 : index2 + 1;

        out.inferOut = gpuOut[index2];
        out.d2is = job.d2is;
        out.inferNum = job.inferNum;

        // 流同步后,获取该batchSize推理结果
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,gpuOut内只有两块空间,因此队列长度只能限定为2
            cv_.wait(l, [&]() { return qOuts.size() < 2; });
            qOuts.push(out);
            cv_.notify_one();
        }

        // 再判断一次推理数据队列,若图片都处理完且队列空,说明推理结束,直接退出线程
        if (queueFinish && qJobs.empty()) break;
    }
    inferFinish = true;
    totalTime2 = timer.timeCount(t);
    printf("infer use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
}

void InferImpl::inferPost(ParamBase &curParam, Infer *curFunc) {
    // 记录后处理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();

    unsigned long mallocSize = this->memory[1] * sizeof(float), singleOutputSize = this->memory[1] / curParam.batchSize;
    batchBoxesType batchBoxes;
    batchBoxesType boxes;
//    传入图片总数, 已处理图片数量
    int totalNum = 0, count = 0;
    bool flag = true;
    Out out;

    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // 队列不为空, 就说明图片已推理好,退出等待,进行后处理. 推理结束,并且队列为空,退出等待,因为此时推理工作已完成
            cv_.wait(l, [&]() { return !qOuts.empty() || (inferFinish && qOuts.empty()); });
            // 退出推理线程
            if (inferFinish && qOuts.empty()) break;
            out = qOuts.front();
            qOuts.pop();
            cv_.notify_one();
            if (flag) {
                totalNum = qfJobLength.front();
                qfJobLength.pop();
                flag = false;
            }
        }


        // 转移到内存中处理
        cudaMemcpy(pinMemoryOut, out.inferOut, mallocSize, cudaMemcpyDeviceToHost);
        curParam.d2is = out.d2is;

        auto qT1 = timer.curTimePoint();
        count += out.inferNum;

        curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, out.inferNum, boxes);
        //将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), boxes.begin(), boxes.end());
        boxes.clear();

        totalTime += timer.timeCount(qT1);

        // 当commit中传入图片处理完时,set value, 返回这批图片的结果. 重新计数, 并返回下一次要输出推理结果的图片数量
        if (totalNum <= count) {
            // 输出解码后的结果,在commit中接收
            fJob1.batchResult->set_value(batchBoxes);
            count = 0;
            flag = true;
        }
    }
    totalTime2 = timer.timeCount(t);
    printf("post  use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);

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

    // 将输入输出所占空间大小返回
    std::vector<int> memory = {inputSize, outputSize};
    return memory;
}

bool InferImpl::startUpThread(ParamBase &param, Infer &curFunc) {
    try {
        preThread = std::make_shared<std::thread>(&InferImpl::inferPre, this, std::ref(param));
        inferThread = std::make_shared<std::thread>(&InferImpl::inferTrt, this, std::ref(param));
        postThread = std::make_shared<std::thread>(&InferImpl::inferPost, this, std::ref(param), &curFunc);
    } catch (std::string &error) {
        printf("pre or infer or post thread start up fail !\n");
        printf("线程启动失败具体错误: %s\n", error.c_str());
        return false;
    }

    printf("pre,infer,post thread start up success !\n");
    return true;
}

InferImpl::InferImpl(std::vector<int> &memory) {
    this->memory = memory;

    cudaStream_t initStream;
    cudaStreamCreate(&initStream);

    cudaStreamCreate(&preStream);
    cudaStreamCreate(&inferStream);
    cudaStreamCreate(&postStream);

    unsigned long InMallocSize = memory[0] * sizeof(float);
    unsigned long OutMallocSize = memory[1] * sizeof(float);

//    checkRuntime(cudaMallocAsync(&gpuMemoryIn0, InMallocSize, initStream));
//    checkRuntime(cudaMallocAsync(&gpuMemoryIn1, InMallocSize, initStream));

//    checkRuntime(cudaMallocAsync(&gpuMemoryOut0, OutMallocSize, initStream));
//    checkRuntime(cudaMallocAsync(&gpuMemoryOut1, OutMallocSize, initStream));

    checkRuntime(cudaMallocHost(&pinMemoryOut, InMallocSize));
    checkRuntime(cudaMallocHost(&pinMemoryIn, OutMallocSize));

    gpuIn[0] = gpuMemoryIn0;
    gpuIn[1] = gpuMemoryIn1;
    gpuOut[0] = gpuMemoryOut0;
    gpuOut[1] = gpuMemoryOut1;

    checkRuntime(cudaStreamDestroy(initStream));
}

InferImpl::~InferImpl() {

    if (workRunning) {
        workRunning = false;
        cv_.notify_all();
    }

    if (preThread->joinable()) preThread->join();
    if (inferThread->joinable()) inferThread->join();
    if (postThread->joinable()) postThread->join();

    checkRuntime(cudaFree(gpuMemoryIn0));
    checkRuntime(cudaFree(gpuMemoryIn1));
    checkRuntime(cudaFree(gpuMemoryOut0));
    checkRuntime(cudaFree(gpuMemoryOut1));

    checkRuntime(cudaFreeHost(pinMemoryIn));
    checkRuntime(cudaFreeHost(pinMemoryOut));

    checkRuntime(cudaStreamDestroy(preStream));
    checkRuntime(cudaStreamDestroy(inferStream));
    checkRuntime(cudaStreamDestroy(postStream));
}

std::shared_ptr<Infer> createInfer(ParamBase &param, const std::string &enginePath, Infer &curFunc) {
//    如果创建引擎不成功就reset
    if (!InferImpl::getEngineContext(param, enginePath)) return nullptr;

    std::vector<int> memory = InferImpl::setBatchAndInferMemory(param);
    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl(memory));

    //若实例化失败
    if (!instance) {
        printf("InferImpl instance fail\n");
        instance.reset();
        return instance;
    }

    // 若线程启动失败,也返回空实例. 所有的错误信息都在函数内部打印
    if (!instance->startUpThread(param, curFunc)) {
        instance.reset();
        return instance;
    }
    return instance;
}