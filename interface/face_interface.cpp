//
// Created by Administrator on 2023/1/9.
//
#include "../utils/general.h"
#include "face_interface.h"


bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line) {
    if (cudaSuccess != code) {
        const char *errName = cudaGetErrorName(code);
        const char *errMsg = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, errName, errMsg);
        return false;
    }
    return true;
}

//使用所有加速算法的初始化部分: 初始化参数,构建engine, 反序列化制作engine
int initCommon(struct ConfigBase &confSpecific, class AlgorithmBase *funcSpecific) {
    //使用从so文件解析出来的函数, 初始化模型参数
    funcSpecific->initParam(&(confSpecific));

    //获取engine绝对路径
    confSpecific.enginePath = AlgorithmBase::getEnginePath(confSpecific);

    std::vector<unsigned char> engineFile;
    // 判断引擎文件是否存在,如果不存在,要先构建engine
    if (std::filesystem::exists(confSpecific.enginePath))
        // engine存在,直接加载engine文件,反序列化引擎文件到内存
        engineFile = AlgorithmBase::loadEngine(confSpecific.enginePath);
    else {
        //engine不存在,先build,序列化engine到硬盘, 再执行反序列化操作
        if (AlgorithmBase::buildEngine(confSpecific.onnxPath, confSpecific.enginePath, confSpecific.batchSize))
            engineFile = AlgorithmBase::loadEngine(confSpecific.enginePath);
    }

    if (engineFile.empty()) return -1;

//   也可以直接从字符串提取名字 confSpecific.enginePath.substr(confSpecific.enginePath.find_last_of("/"),-1)
    std::cout << "start create engine: " << std::filesystem::path(confSpecific.enginePath).filename() << std::endl;
    confSpecific.engine = AlgorithmBase::createEngine(engineFile);
    if (nullptr == confSpecific.engine) {
        std::cout << "failed engine name: " << std::filesystem::path(confSpecific.enginePath).filename() << std::endl;
        return -1;
    }
    std::cout << "create engine success: " << std::filesystem::path(confSpecific.enginePath).filename() << std::endl;
    return 0;
}

int initEngine(struct productConfig &conf, struct productFunc &func) {
    //人脸检测模型初始化
    if (nullptr == func.yoloFace) {
        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtFaceYolo.so");
        if (!curAlg) printf("error");

        // 把函数指针从init函数中提出来,在infer推理阶段使用.
        func.yoloFace = curAlg;

        int initFlag = initCommon(conf.yoloConfig, func.yoloFace);
        if (0 > initFlag) {
            printf("yolo face init failed\n");
            return -1;
        }
    }

    // 其他检测模型初始化
    if (nullptr == func.yoloDetect) {
        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtFaceYolo.so");
        if (!curAlg) printf("error");

        func.yoloDetect = curAlg;
        int initFlag = initCommon(conf.detectConfig, func.yoloDetect);
        if (0 > initFlag) {
            printf("yolo detect init failed\n");
            return -1;
        }
    }

    return 0;
}

//设置推理过程中,输入输出tensor在内存,显存上使用空间
std::vector<int> setMemorySize(struct ConfigBase &confSpecific, float *pinInput, float *pinOutput, float *gpuInput, float *gpuOutput) {
    //计算输入tensor所占存储空间大小
    auto inputShape = confSpecific.engine->getTensorShape(confSpecific.inputName.c_str());
    int inputSize = confSpecific.batchSize * inputShape.d[1] * inputShape.d[2] * inputShape.d[3];

    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
    checkRuntime(cudaMallocHost(&pinInput, inputSize * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuInput, inputSize * sizeof(float)));

    // 获得输出tensor形状,计算输出所占存储空间
    auto outputShape = confSpecific.engine->getTensorShape(confSpecific.outputName.c_str());
    int outputSize = confSpecific.batchSize * outputShape.d[1] * outputShape.d[2];

    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
    checkRuntime(cudaMallocHost(&pinOutput, outputSize * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuOutput, outputSize * sizeof(float)));
//    std::tuple<int,nvinfer1::Dims32> aa;
    std::vector<int> res(inputSize,outputSize);

    return res;
}

//使用引擎推理图片
//int inferEngine(Handle engine, unsigned char *imgData, int imgWidth, int imgHeight, int min_face_size, int mode,
//                int imgPixelFormat, int &res_num) {
//int inferEngine(struct productConfig &conf, struct productFunc &func, cv::Mat &image, struct productOutput out) {
int inferEngine_back(struct productConfig &conf, struct productFunc &func, std::vector<cv::Mat> &matVector,
                struct productOutput out) {

//    func.yoloFace->preProcess(image);

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();
    //
    float *pinInput = nullptr, *pinOutput = nullptr, *gpuInput = nullptr, *gpuOutput = nullptr;
//    //计算输入tensor所占存储空间大小
//    int inputSize = conf.detectConfig.batchSize * inputChannel * inputHeight * inputWidth;
//
//    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
//    checkRuntime(cudaMallocHost(&pinInput, inputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuInput, inputSize * sizeof(float)));

    // 填充灰边, 缩放图片到模型输入指定的尺寸
    cv::Mat scaleImage = letterBox(image, inputWidth, inputHeight, d2i1);

//    BGR2RGB(scaleImage, pinInput);

    //模型输入数据从锁页内存转到gpu上
    checkRuntime(cudaMemcpyAsync(gpuInput, pinInput, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));

    //获得输入tensor形状,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = conf.detectConfig.engine->getTensorShape(conf.detectConfig.inputName.c_str());
    inputShape.d[0] = conf.detectConfig.batchSize;
    context->setInputShape(conf.detectConfig.inputName.c_str(), inputShape);

//    // 获得输出tensor形状,计算输出所占存储空间
//    auto outputShape = conf.detectConfig.engine->getTensorShape(conf.detectConfig.outputName.c_str());
////    int boxNum = outputShape.d[1];
////    int predictNum = outputShape.d[2];
////    int classNum = predictNum - 5;
//    int outputSize = conf.detectConfig.batchSize * outputShape.d[1] * outputShape.d[2];
//
//    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinOutput, outputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuOutput, outputSize * sizeof(float)));


    // 指定onnx中输入输出tensor名
    context->setTensorAddress(conf.detectConfig.inputName.c_str(), gpuInput);
    context->setTensorAddress(conf.detectConfig.outputName.c_str(), gpuOutput);

    // 执行异步推理
    context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinOutput, gpuOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);

    return 0;
}

int inferEngine(struct productConfig &conf, struct productFunc &func, std::vector<cv::Mat> &matVector,
                     struct productOutput out) {

//    func.yoloFace->preProcess(image);

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();

    float *pinInput = nullptr, *pinOutput = nullptr, *gpuInput = nullptr, *gpuOutput = nullptr;
    std::vector<int> res =setMemorySize(conf.detectConfig, pinInput, pinOutput, gpuInput, gpuOutput);


    // 填充灰边, 缩放图片到模型输入指定的尺寸
    cv::Mat scaleImage = letterBox(image, inputWidth, inputHeight, d2i1);

//    BGR2RGB(scaleImage, pinInput);

    //模型输入数据从锁页内存转到gpu上
    checkRuntime(cudaMemcpyAsync(gpuInput, pinInput, res[0] * sizeof(float), cudaMemcpyHostToDevice, stream));

    //获得输入tensor形状,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = conf.detectConfig.engine->getTensorShape(conf.detectConfig.inputName.c_str());
    inputShape.d[0] = conf.detectConfig.batchSize;
    context->setInputShape(conf.detectConfig.inputName.c_str(), inputShape);

    // 指定onnx中输入输出tensor名
    context->setTensorAddress(conf.detectConfig.inputName.c_str(), gpuInput);
    context->setTensorAddress(conf.detectConfig.outputName.c_str(), gpuOutput);

    // 执行异步推理
    context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinOutput, gpuOutput, res[1] * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);

    return 0;
}

int
inferEngine(struct productConfig &conf, struct productFunc &func, std::vector<cv::cuda::GpuMat> &images, int &res_num) {


    return 0;
}
