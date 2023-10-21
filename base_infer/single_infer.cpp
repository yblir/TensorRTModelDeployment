//
// Created by Administrator on 2023/4/1.
//
#include <cuda_runtime.h>
//onnx解释器头文件
#include <NvOnnxParser.h>
#include <fstream>

#include "../utils/general.h"
#include "../utils/box_utils.h"
#include "../builder/trt_builder.h"

#include "infer.h"

class InferImpl : public Infer {
public:
    explicit InferImpl(std::vector<int> &memory);
    ~InferImpl() override;
    // 创建推理engine
    static bool getEngineContext(BaseParam &curParam);
    static std::vector<int> setBatchAndInferMemory(BaseParam &curParam);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
//    preProcess,postProcess空实现,具体实现由实际继承Infer.h的应用完成
    int preProcess(BaseParam &param, cv::Mat &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam &param, pybind11::array &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam &param, std::vector<pybind11::array> &images, float *pinMemoryCurrentIn) override {};

    int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};

//    具体应用调用commit方法,推理数据传入队列, 直接返回future对象. 数据依次经过trtPre,trtInfer,trtPost三个线程,结果通过future.get()获得
    batchBoxesType commit(BaseParam *param, const InputData *data) override;
    batchBoxesType commit(BaseParam *param, const pybind11::array &img) override;
private:
    float *gpuIn = nullptr, *pinMemoryIn = nullptr;
    float *gpuOut = nullptr, *pinMemoryOut = nullptr;
    std::vector<int> memory;
    // 读取从路径读入的图片矩阵
    cv::Mat mat;

    futureJob fJob;
    std::queue<futureJob> qfJobs;
    batchBoxesType batchBoxes, batchBox;

    //创建cuda任务流,对应上述三个处理线程
    cudaStream_t commitStream{};
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Timer timer = Timer();

//使用所有加速算法的初始化部分: 初始化参数,构建engine, 反序列化制作engine
bool InferImpl::getEngineContext(BaseParam &curParam) {
    std::vector<unsigned char> engine;
    try {
        engine = TRT::getEngine(curParam);
    } catch (std::string &error) {
//        捕获未知的异常情况
        logError("load engine failure: %s", error.c_str());
        return false;
    }
    if (engine.empty()) {
        logError("load engine failure");
        return false;
    }
//   也可以直接从字符串提取名字 curParam.enginePath.substr(curParam.enginePath.find_last_of("/"),-1)
    logSuccess("load engine success: %s", std::filesystem::path(curParam.enginePath).filename().c_str());

    // 创建engine并获得执行上下文context =======================================================================================
    TRTLogger logger;
//   在trt8.6版本中,使用自动释放会出错
//    auto runtime = ptrFree(nvinfer1::createInferRuntime(logger));
    auto runtime = nvinfer1::createInferRuntime(logger);

    try {
        curParam.engine = ptrFree(runtime->deserializeCudaEngine(engine.data(), engine.size()));
    } catch (std::string &error) {
//        捕获未知的异常情况
        logSuccess("deserialize cuda engine failure: %s", error.c_str());
        return false;
    }
    if (nullptr == curParam.engine) {
        logError("deserialize cuda engine failure");
        return false;
    }
    logSuccess("deserialize cuda engine success");

//    限定推理引擎只有1个输入和1个输出,不处理多节点模型.是因为有1个是输入.剩下的才是输出,所以要-1
    if (2 != curParam.engine->getNbIOTensors()) {
        logError("expect engine's node num is 2, 1 input and 1 output, but now node num is %d", curParam.engine->getNbIOTensors() - 1);
        return false;
    }

    curParam.context = ptrFree(curParam.engine->createExecutionContext());
    logSuccess("create context success");
    return true;
}

// 不再用结构体InputData管理输入数据. 同时std::vector<pybind11::array>中, vector不能用清空,内存会一直增加
batchBoxesType InferImpl::commit(BaseParam *param, const pybind11::array &img) {
// Infer &curFunc,
    batchBoxes.clear();
    unsigned long inMallocSize = this->memory[0] * sizeof(float);
    unsigned long outMallocSize = this->memory[1] * sizeof(float);
    // count统计推理图片数量,最后一次,可能小于batchSize.imgNums统计预处理图片数量,index是gpuIn的索引,两个显存轮换使用
    int countPre = 0, index = 0, inferNum = 0, singleInputSize = this->memory[0] / param->batchSize;

    param->context->setTensorAddress(param->inputName.c_str(), gpuIn);
    param->context->setTensorAddress(param->outputName.c_str(), gpuOut);

    checkRuntime(cudaMemcpyAsync(gpuIn, img.data(0), inMallocSize, cudaMemcpyHostToDevice, commitStream));

    param->context->enqueueV3(commitStream);

    cudaMemcpyAsync(pinMemoryOut, gpuOut, outMallocSize, cudaMemcpyDeviceToHost, commitStream);
    checkRuntime(cudaStreamSynchronize(commitStream));

    batchBox.push_back(*pinMemoryOut);
    batchBox.push_back(*(pinMemoryOut + 1));
//        logInfo("post Process success");
    //将每次后处理结果合并到输出vector中
    batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());
    batchBox.clear();

    return batchBoxes;
}


std::vector<int> InferImpl::setBatchAndInferMemory(BaseParam &curParam) {
    //计算输入tensor所占存储空间大小,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = curParam.engine->getTensorShape(curParam.inputName.c_str());
//    std::cout << "inputshape = " << inputShape.d[0] << " " << inputShape.d[1] << " " << inputShape.d[2] << " " << inputShape.d[3]
//              << std::endl;
    inputShape.d[0] = curParam.batchSize;
//    printf("000\n");

    curParam.context->setInputShape(curParam.inputName.c_str(), inputShape);
//    printf("11111\n");
    //batchSize * c * h * w
    int inputSize = curParam.batchSize * inputShape.d[1] * inputShape.d[2] * inputShape.d[3];

    // 获得输出tensor形状,计算输出所占存储空间
    auto outputShape = curParam.engine->getTensorShape(curParam.outputName.c_str());
//    std::cout << "outshape = " << outputShape.d[0] << " " << outputShape.d[1] << " " << outputShape.d[2] << std::endl;
//    printf("222\n");
    // 记录这两个输出维度数量,在后处理时会用到
//    curParam.predictNums = outputShape.d[1];
//    curParam.predictLength = outputShape.d[2];
    // 计算推理结果所占内存空间大小
    int outputSize = curParam.batchSize * outputShape.d[1];
//    int outputSize = batchSize
//    printf("333\n");
    // 将batchSize个输入输出所占空间大小返回
    std::vector<int> memory = {inputSize, outputSize};
//    printf("444\n");
    return memory;
}

//bool InferImpl::startThread(BaseParam &curParam, Infer &curFunc) {
//    try {
//        preThread = std::make_shared<std::thread>(&InferImpl::trtPre, this, std::ref(curParam), &curFunc);
//        inferThread = std::make_shared<std::thread>(&InferImpl::trtInfer, this, std::ref(curParam));
//        postThread = std::make_shared<std::thread>(&InferImpl::trtPost, this, std::ref(curParam), &curFunc);
//    } catch (std::string &error) {
//        logError("thread start failure: %s !", error.c_str());
//        return false;
//    }
//
//    logSuccess("thread start success !");
//    return true;
//}

InferImpl::InferImpl(std::vector<int> &memory) {
    this->memory = memory;

    cudaStream_t initStream;
    cudaStreamCreate(&initStream);

//    cudaStreamCreate(&preStream);
//    cudaStreamCreate(&inferStream);
//    cudaStreamCreate(&postStream);

    unsigned long InMallocSize = memory[0] * sizeof(float);
    unsigned long OutMallocSize = memory[1] * sizeof(float);

    checkRuntime(cudaMallocAsync(&gpuIn, InMallocSize, initStream));
//    checkRuntime(cudaMallocAsync(&gpuMemoryIn1, InMallocSize, initStream));

    checkRuntime(cudaMallocAsync(&gpuOut, OutMallocSize, initStream));
//    checkRuntime(cudaMallocAsync(&gpuMemoryOut1, OutMallocSize, initStream));

    checkRuntime(cudaMallocHost(&pinMemoryIn, InMallocSize));
    checkRuntime(cudaMallocHost(&pinMemoryOut, OutMallocSize));

//    gpuIn[0] = gpuMemoryIn0;
//    gpuIn[1] = gpuMemoryIn1;
//    gpuOut[0] = gpuMemoryOut0;
//    gpuOut[1] = gpuMemoryOut1;

    checkRuntime(cudaStreamDestroy(initStream));
}

InferImpl::~InferImpl() {
    logInfo("start executing destructor ...");
//    if (workRunning) {
//        workRunning = false;
//        cv_.notify_all();
//    }

//    if (preThread->joinable()) preThread->join();
//    if (inferThread->joinable()) inferThread->join();
//    if (postThread->joinable()) postThread->join();

    checkRuntime(cudaFree(gpuIn));
//    checkRuntime(cudaFree(gpuMemoryIn1));
    checkRuntime(cudaFree(gpuOut));
//    checkRuntime(cudaFree(gpuMemoryOut1));

    checkRuntime(cudaFreeHost(pinMemoryIn));
    checkRuntime(cudaFreeHost(pinMemoryOut));

//    checkRuntime(cudaStreamDestroy(preStream));
//    checkRuntime(cudaStreamDestroy(inferStream));
//    checkRuntime(cudaStreamDestroy(postStream));
//    printf("析构函数\n");
}

std::shared_ptr<Infer> createInfer(BaseParam &curParam, Infer &curFunc) {
//    如果创建引擎不成功就reset
    if (!InferImpl::getEngineContext(curParam)) {
//        logError("getEngineContext Fail");
        return nullptr;
    }
    std::vector<int> memory;
    try {
        memory = InferImpl::setBatchAndInferMemory(curParam);
    } catch (std::string &error) {
        logError("setBatchAndInferMemory failure: %s !", error.c_str());
        return nullptr;
    }
//    logInfo("setbatch ok");
    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl(memory));

    //若实例化失败 或 若线程启动失败,也返回空实例. 所有的错误信息都在函数内部打印
    if (!instance) {
//        logError("InferImpl instance Fail");
        instance.reset();
        return nullptr;
    }

    return instance;
}