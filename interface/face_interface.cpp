//
// Created by Administrator on 2023/1/9.
//

#include "face_interface.h"

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

//使用引擎推理图片
//int inferEngine(Handle engine, unsigned char *imgData, int imgWidth, int imgHeight, int min_face_size, int mode,
//                int imgPixelFormat, int &res_num) {
int inferEngine(struct productConfig &conf, struct productFunc &func, std::vector<cv::Mat> &images, int &res_num) {
    func.yoloFace->preProcess(images);

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();

    float *pinInput = nullptr, *gpuInput = nullptr;
    float *pinOutput = nullptr, *gpuOutput = nullptr;

    // 指定onnx中输入输出tensor名, 将tensor直接传输给对应输入输出名
    context->setTensorAddress(conf.detectConfig.inputName.c_str(), gpuInput);
    context->setTensorAddress(conf.detectConfig.outputName.c_str(), gpuOutput);

    // 执行异步推理
    context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinOutput, gpuOutput, outputNumel * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);

    return 0;
}

int
inferEngine(struct productConfig &conf, struct productFunc &func, std::vector<cv::cuda::GpuMat> &images, int &res_num) {


    return 0;
}
