//
// Created by Administrator on 2023/4/1.
//

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
/* 弃用引用形式传参
    preProcess,postProcess空实现,具体实现由实际继承Infer.h的应用完成
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override {};
    int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam &param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) override {};
    int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};
*/
    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override {};

    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};
    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, std::map<int, imgBoxesType> &result,
                    const int &index) override {};

#ifdef PYBIND11
    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) override {};
#endif

    std::vector<int> getMemory() override;
    std::vector<int> memory;
};

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

    // 创建engine并获得执行上下文context
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
    // if (2 != curParam.engine->getNbIOTensors()) {
    //     logError("expect engine's node num is 2, 1 input and 1 output, but now node num is %d", curParam.engine->getNbIOTensors() - 1);
    //     return false;
    // }

//   模型至少有1个输入和1个输出
    if (2 > curParam.engine->getNbIOTensors()) {
        logError("expect engine's node num not less 2, but now node num is %d", curParam.engine->getNbIOTensors());
        return false;
    }

    curParam.context = ptrFree(curParam.engine->createExecutionContext());
    logSuccess("create context success");
    return true;
}

// 计算当前用于推理的输入输出大小,和配置推理地址
std::vector<int> InferImpl::getMemory() {

    return this->memory;
}

//// 不再用结构体InputData管理输入数据. 同时std::vector<pybind11::array>中, vector不能用清空,内存会一直增加
//batchBoxesType InferImpl::commit(BaseParam *param, const pybind11::array &img) {
////  类型强转, 然后才能调用子类中具体算法前后处理, 这样会破坏通用型,现在还没想到更好处理方法
////    auto curParam=reinterpret_cast<
//// Infer &curFunc,
//    batchBoxes.clear();
//
//    int a = preProcess(img, pinMemoryIn);
//    checkRuntime(cudaMemcpyAsync(gpuIn, img.data(0), trtInSize, cudaMemcpyHostToDevice, commitStream));
//
//    param->context->enqueueV3(commitStream);
//
//    cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutSize, cudaMemcpyDeviceToHost, commitStream);
//    checkRuntime(cudaStreamSynchronize(commitStream));
//
//    batchBox.push_back(*pinMemoryOut);
//    batchBox.push_back(*(pinMemoryOut + 1));
//
//    //将每次后处理结果合并到输出vector中
//    batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());
//    batchBox.clear();
//
//    return batchBoxes;
//}

std::vector<int> InferImpl::setBatchAndInferMemory(BaseParam &curParam) {
    int inputSize = 1, outputSize = 1;
    //计算输入tensor所占存储空间大小,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = curParam.engine->getTensorShape(curParam.inputName.c_str());
    // 获得输出tensor形状,计算输出所占存储空间
    auto outputShape = curParam.engine->getTensorShape(curParam.outputName.c_str());

//    重置batchsize,以外部输入batch为准
    inputShape.d[0] = curParam.batchSize;
    outputShape.d[0] = curParam.batchSize;

//    设定trt engine输入输出shape
    curParam.trtInputShape = inputShape;
    curParam.trtOutputShape = outputShape;

    curParam.context->setInputShape(curParam.inputName.c_str(), inputShape);

//    计算batchsize个输入输出占用空间大小, inputSize=batchSize * c * h * w
    for (int i = 0; i < inputShape.nbDims; ++i) {
        inputSize *= inputShape.d[i];
    }
//    outputSize=batchSize*...
    for (int i = 0; i < outputShape.nbDims; ++i) {
        outputSize *= outputShape.d[i];
    }

    // 将batchSize个输入输出所占空间大小返回
    std::vector<int> memory = {inputSize, outputSize};
    return memory;
}


InferImpl::InferImpl(std::vector<int> &memory) {
    this->memory = memory;

}

InferImpl::~InferImpl() {}

std::shared_ptr<Infer> createInfer(Infer &curFunc, BaseParam &curParam) {
//   根据外部传入gpu编号设置工作gpu
    int nGpuNum = 0;
    checkRuntime(cudaGetDeviceCount(&nGpuNum));
    if (0 == nGpuNum) {
        logError("Current device does not detect GPU");
        return nullptr;
    }
    if (curParam.gpuId >= nGpuNum) {
        logError("GPU device setting failure, max GPU index = %d, but set GPU Id = %d", nGpuNum - 1, curParam.gpuId);
        return nullptr;
    }
    checkRuntime(cudaSetDevice(curParam.gpuId));

//    如果创建引擎不成功就reset
    if (!InferImpl::getEngineContext(curParam)) {
        return nullptr;
    }
    std::vector<int> memory;
    try {
        memory = InferImpl::setBatchAndInferMemory(curParam);
    } catch (std::string &error) {
        logError("setBatchAndInferMemory failure: %s !", error.c_str());
        return nullptr;
    }

    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl(memory));

    //若实例化失败 或 若线程启动失败,也返回空实例. 所有的错误信息都在函数内部打印
    if (!instance) {
        instance.reset();
        return nullptr;
    }

    return instance;
}