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
int initCommon(struct parmBase &curParm, class AlgorithmBase *curFunc) {
    TRTLogger logger;
    //使用从so文件解析出来的函数, 初始化模型参数
    curFunc->initParam(&(curParm));

    //获取engine绝对路径
    curParm.enginePath = AlgorithmBase::getEnginePath(curParm);

    std::vector<unsigned char> engineFile;
    // 判断引擎文件是否存在,如果不存在,要先构建engine
    if (std::filesystem::exists(curParm.enginePath))
        // engine存在,直接加载engine文件,反序列化引擎文件到内存
        engineFile = AlgorithmBase::loadEngine(curParm.enginePath);
    else {
        //engine不存在,先build,序列化engine到硬盘, 再执行反序列化操作
        if (AlgorithmBase::buildEngine(curParm.onnxPath, curParm.enginePath, curParm.batchSize))
            engineFile = AlgorithmBase::loadEngine(curParm.enginePath);
    }

    if (engineFile.empty()) return -1;

//   也可以直接从字符串提取名字 curParm.enginePath.substr(curParm.enginePath.find_last_of("/"),-1)
    std::cout << "start create engine: " << std::filesystem::path(curParm.enginePath).filename() << std::endl;
//    AlgorithmBase::createEngine(engineFile,curParm.engine);

    // 创建引擎并获得执行上下文========================================================
    auto runtime = prtFree(nvinfer1::createInferRuntime(logger));
    curParm.engine = prtFree(runtime->deserializeCudaEngine(engineFile.data(), engineFile.size()));

    if (nullptr == curParm.engine) {
        printf("deserialize cuda engine failed.\n");
        return -1;
    }

//     engine->getNbIOTensors()-1,是因为有1个是输入.剩下的才是输出
    if (2 != curParm.engine->getNbIOTensors()) {
        printf("create engine failed: onnx导出有问题,必须是1个输入和1个输出,当前有%d个输出\n", curParm.engine->getNbIOTensors() - 1);
        return -1;
    }

//    curParm.engine->createExecutionContext();
    if (nullptr == curParm.engine) {
        std::cout << "failed engine name: " << std::filesystem::path(curParm.enginePath).filename() << std::endl;
        return -1;
    }
    curParm.context = prtFree(curParm.engine->createExecutionContext());
    std::cout << "create engine success: " << std::filesystem::path(curParm.enginePath).filename() << std::endl;

    return 0;
}

int initEngine(struct productConfig &conf, struct productFunc &func) {
    //人脸检测模型初始化
//    if (nullptr == func.yoloFace) {
//        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
//                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtFaceYolo.so");
//        if (!curAlg) printf("error");
//
//        // 把函数指针从init函数中提出来,在infer推理阶段使用.
//        func.yoloFace = curAlg;
//
//        int initFlag = initCommon(conf.yoloConfig, func.yoloFace);
//        if (0 > initFlag) {
//            printf("yolo face init failed\n");
//            return -1;
//        }
//    }

    // 其他检测模型初始化
    if (nullptr == func.yoloDetect) {
        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtYoloDetect.so");
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

// setBatchAndInferMemory
//设置推理过程中,输入输出tensor在内存,显存上使用存储空间大小.并返回输入tensor shape和输入输出空间大小值
std::vector<int> setBatchAndInferMemory(struct parmBase &curParm,
                                        const std::shared_ptr<nvinfer1::IExecutionContext> &context,
                                        float *pinMemoryIn, float *pinMemoryOut, float *gpuMemoryIn, float *gpuMemoryOut) {
    //计算输入tensor所占存储空间大小,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = curParm.engine->getTensorShape(curParm.inputName.c_str());
    inputShape.d[0] = curParm.batchSize;
    context->setInputShape(curParm.inputName.c_str(), inputShape);
    //batchSize * c * h * w
    int inputSize = curParm.batchSize * inputShape.d[1] * inputShape.d[2] * inputShape.d[3];

//    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
//    checkRuntime(cudaMallocHost(&pinMemoryIn, inputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryIn, inputSize * sizeof(float)));

    // 获得输出tensor形状,计算输出所占存储空间
    auto outputShape = curParm.engine->getTensorShape(curParm.outputName.c_str());
    int outputSize = curParm.batchSize * outputShape.d[1] * outputShape.d[2];

//    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, outputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryOut, outputSize * sizeof(float)));

    // 使用元组返回多个多个不同的类型值, 供函数外调用,有以下两种方式可使用
//    return std::make_tuple(inputShape, inputSize, outputSize);
//    return std::tuple<nvinfer1::Dims32, int, int>{inputShape, inputSize, outputSize};

    // 将输入输出所占空间大小返回
    std::vector<int> memory = {inputSize, outputSize};
    return memory;
}

// 适用于模型推理的通用trt流程
int trtEnqueueV3(struct parmBase &curParm, std::vector<int> memory,
                 float *pinMemoryIn, float *pinMemoryOut, float *gpuMemoryIn, float *gpuMemoryOut) {
    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //模型输入数据从锁页内存转到gpu上
    checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn, memory[0] * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 指定onnx中输入输出tensor名
    curParm.context->setTensorAddress(curParm.inputName.c_str(), gpuMemoryIn);
    curParm.context->setTensorAddress(curParm.outputName.c_str(), gpuMemoryOut);
    // 执行异步推理
    curParm.context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinMemoryOut, gpuMemoryOut, memory[1] * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);
}

// 通用推理接口
int trtInferProcess(struct parmBase &curParm, class AlgorithmBase *curFunc, std::vector<cv::Mat> matVector) {
    //配置锁页内存,gpu显存指针
    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
    //0:当前推理模型输入tensor存储空间大小,1:当前推理输出结果存储空间大小
    std::vector<int> memory = setBatchAndInferMemory(curParm, curParm.context, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);

    // 以下,开辟内存操作不能在单独函数中完成,因为是二级指针,在当前函数中开辟内存,离开函数内存空间会消失
    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
    checkRuntime(cudaMallocHost(&pinMemoryIn, memory[0] * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
    checkRuntime(cudaMallocHost(&pinMemoryOut, memory[1] * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuMemoryOut, memory[1] * sizeof(float)));

    // 预处理,一次处理batchSize张图片, 包括尺寸缩放,归一化,色彩转换,图片数据从内存提取到gpu
    int count = 0;
    // 计算单张图片占用空间
    int singleSize = memory[0] / curParm.batchSize;
    //存储要推理的一个batchSize的图片,用于后处理操作
    std::vector<cv::Mat> images;

    for (auto &mat: matVector) {
        count += 1;
        images.emplace_back(mat);

        // 遍历所有图片,若图片数量不够一个batch,加入的处理队列中
        if (count <= curParm.batchSize)
            // 处理单张图片,每次预处理图片,指针要跳过前面处理过的图片
            curFunc->preProcess(mat, pinMemoryIn + count * singleSize, curParm);

        //够一个batchSize,执行推理. 或者当循环vector取到最后一个元素时(当前元素地址与最后一个元素地址相同),不论是否够一个batchSize, 都要执行推理
        if (count == curParm.batchSize || &mat == &matVector.back()) {
            trtEnqueueV3(curParm, memory, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);
            //后处理,处理一个batchSize的所有图片
            curFunc->postProcess(images, pinMemoryOut, curParm);
            // 清空标记,重新开始下一个bachSize.
            count = 0;
            images.clear();
        }
    }
}

int inferEngine(struct productConfig &conf, struct productFunc &func,
                std::vector<cv::Mat> &matVector, struct productOutput out) {
    // 以engine是否存在为判定,存在则执行推理
    if (nullptr != conf.detectConfig.engine) trtInferProcess(conf.detectConfig, func.yoloDetect, matVector);

//    if (nullptr != conf.yoloConfig.engine) trtInferProcess(conf.yoloConfig, func.yoloFace, matVector);

    return 0;
}
