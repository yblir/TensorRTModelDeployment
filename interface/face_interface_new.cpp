//
// Created by Administrator on 2023/1/9.
//
#include <cuda_runtime.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <future>
#include <condition_variable>
#include <future>

#include "../utils/general.h"
#include "face_interface_new.h"
#include "../utils/box_utils.h"

//struct
std::mutex lock_;

//存储与q_mats队列中对应图片的图片缩放参数
//std::queue<std::vector<float>> q_d2is;

std::condition_variable condition;

// 推理输入
struct Job {
    std::shared_ptr<std::promise<float *>> gpuOutputPtr;
    float *inputTensor;

};

// 推理输出
struct Out {
    std::shared_future<float *> inferResult;
    std::vector<std::vector<float>> d2is;
    int inferNum;
};

//std::queue<float *> qJobs;
std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
std::queue<Out> qOuts;

std::atomic<bool> queueFinish{false};
std::atomic<bool> inferFinish{false};
std::atomic<bool> batchCopyOk{false};

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
int initCommon(ParamBase &curParam, class AlgorithmBase *curFunc) {
    //获取engine绝对路径
    curParam.enginePath = AlgorithmBase::getEnginePath(curParam);

    std::vector<unsigned char> engineFile;
    // 判断引擎文件是否存在,如果不存在,要先构建engine
    if (std::filesystem::exists(curParam.enginePath))
        // engine存在,直接加载engine文件,反序列化引擎文件到内存
        engineFile = AlgorithmBase::loadEngine(curParam.enginePath);
    else {
        //engine不存在,先build,序列化engine到硬盘, 再执行反序列化操作
        if (AlgorithmBase::buildEngine(curParam.onnxPath, curParam.enginePath, curParam.batchSize))
            engineFile = AlgorithmBase::loadEngine(curParam.enginePath);
    }

    if (engineFile.empty()) return -1;

//   也可以直接从字符串提取名字 curParam.enginePath.substr(curParam.enginePath.find_last_of("/"),-1)
    std::cout << "start create engine: " << std::filesystem::path(curParam.enginePath).filename() << std::endl;

    // 创建engine并获得执行上下文context =======================================================================================
    TRTLogger logger;
    auto runtime = prtFree(nvinfer1::createInferRuntime(logger));
    curParam.engine = prtFree(runtime->deserializeCudaEngine(engineFile.data(), engineFile.size()));

    if (nullptr == curParam.engine) {
        printf("deserialize cuda engine failed.\n");
        return -1;
    }
//    是因为有1个是输入.剩下的才是输出
    if (2 != curParam.engine->getNbIOTensors()) {
        printf("create engine failed: onnx导出有问题,必须是1个输入和1个输出,当前有%d个输出\n",
               curParam.engine->getNbIOTensors() - 1);
        return -1;
    }
    if (nullptr == curParam.engine) {
        std::cout << "failed engine name: " << std::filesystem::path(curParam.enginePath).filename() << std::endl;
        return -1;
    }
    curParam.context = prtFree(curParam.engine->createExecutionContext());
    std::cout << "create engine and context success: " << std::filesystem::path(curParam.enginePath).filename()
              << std::endl;

    return 0;
}

//设置推理过程中,输入输出tensor在内存,显存上使用存储空间大小.并返回输入tensor shape和输入输出空间大小值
std::vector<int> setBatchAndInferMemory(ParamBase &curParam) {
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

// 适用于模型推理的通用trt流程
int trtEnqueueV3(ParamBase &curParam, int &count, float *gpuMemoryOut, cudaStream_t &stream) {
    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            condition.wait(l, [&]() {
                // false, 表示继续等待. true, 表示不等待,跳出wait
                // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理
                // 当图片都处理完,并且队列为空,有要退出等待,因为此时推理工作已完成
                return !qJobs.empty() || (queueFinish && qJobs.empty());
            });
            // 退出推理线程
            if (queueFinish && qJobs.empty()) break;

            auto job = qJobs.front();
            qJobs.pop();
            curParam.context->setTensorAddress(curParam.inputName.c_str(), job.inputTensor);
            // 执行异步推理
            curParam.context->enqueueV3(stream);

            // 消费掉一个,就可以通知队列跳出等待,继续生产
            condition.notify_one();

            // 流同步
            cudaStreamSynchronize(stream);

            // 流同步后,获取该batchSize推理结果
            job.gpuOutputPtr->set_value(gpuMemoryOut);

            // 释放已推理完的显存空间
            cudaFree(job.inputTensor);

        }
    }
    inferFinish = true;
}

//// 通用推理接口
//int trtInferProcess(ParamBase &curParam, AlgorithmBase *curFunc,
//                    std::vector<cv::Mat> &mats, std::vector<std::vector<std::vector<float>>> &result) {
//    //配置锁页内存,gpu显存指针
//    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
//    //0:当前推理模型输入tensor存储空间大小,1:当前推理输出结果存储空间大小
//    std::vector<int> memory = setBatchAndInferMemory(curParam);
//
//    // 以下,开辟内存操作不能在单独函数中完成,因为是二级指针,在当前函数中开辟内存,离开函数内存空间会消失
//    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
//    checkRuntime(cudaMallocHost(&pinMemoryIn, memory[0] * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
//    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, memory[1] * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryOut, memory[1] * sizeof(float)));
//
//    // 预处理,一次处理batchSize张图片, 包括尺寸缩放,归一化,色彩转换,图片数据从内存提取到gpu
//    int count = 0;
//    // 计算模型推理时,单个输入输出tensor占用空间
//    int singleInputSize = memory[0] / curParam.batchSize;
//    int singleOutputSize = memory[1] / curParam.batchSize;
//
//    // 取得最后一个元素的地址
//    auto lastAddress = &mats.back();
//
//    for (auto &mat: mats) {
//        // 记录已推理图片数量
//        count += 1;
//
//        // 遍历所有图片,若图片数量不够一个batch,加入的处理队列中
//        if (count <= curParam.batchSize)
//            // todo 在gpu上开辟两块gpuMemoryIn内存,一个batch预处理数据存储一个memory中,队列长度为2,只有当其中一个memory被取走推理时,
//            // todo 才继续预处理,写入空的memory,存入队列
//            // todo 或者开辟一段有两个batch空间的gpuMemoryIn,持续写入,直到为满,当取走一个baith图片做推理时,再继续写入
//            // 处理单张图片,每次预处理图片,指针要跳过前面处理过的图片
//            curFunc->preProcess(curParam, mat, pinMemoryIn + (count - 1) * singleInputSize);
//
//        //够一个batchSize,执行推理. 或者当循环vector取到最后一个元素时(当前元素地址与最后一个元素地址相同),不论是否够一个batchSize, 都要执行推理
//        if (count == curParam.batchSize || &mat == lastAddress) {
//            //通用推理过程,推理成功后将结果从gpu复制到锁页内存pinMemoryOut
//            trtEnqueueV3(curParam, memory, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);
//            //后处理,函数内循环处理一个batchSize的所有图片
//            curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, count, result);
//
//            // 清0标记,清空用于后处理的images,清空用于图像尺寸缩放的d2is,重新开始下一个bachSize.
//            count = 0;
//            std::vector<std::vector<float>>().swap(curParam.d2is);
//        }
//    }
//}


//void preProcess(ParamBase &curParam, std::vector<std::string> &imgPaths) {
//    auto lastAddress = &imgPaths.back();
//
//    for (auto &imgPath: imgPaths) {
//        // 记录队列最后一个
//        if (&imgPath == lastAddress) queueFinish = true;
//        Job job;
//        // auto stream = cv::cuda::StreamAccessor::wrapStream(m_stream);
//        cv::Mat mat = cv::imread(imgPath);
//        float d2i[6];
//        cv::Mat scaleImage = letterBox(mat, 640, 640, d2i);
//
//        cv::cuda::GpuMat gpuMat(scaleImage);
//        cv::cuda::cvtColor(mat, gpuMat, cv::COLOR_BGR2RGB);
//        // 数值归一化
//        gpuMat.convertTo(gpuMat, CV_32FC3, 1. / 255., 0);
//
//        // 依次存储一个batchSize中图片放射变换参数
//        job.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
//        job.inputTensor = gpuMat;
//
//        {
//            std::unique_lock<std::mutex> l(lock_);
//            condition.wait(l, [&]() {
//                // false, 表示继续等待.
//                // true, 表示不等待,跳出wait
////                wait流程: 一旦进入wait,解锁. 退出wait,又加锁
//                return qJobs.size() < 2 * curParam.batchSize;
//            });
//
//            qJobs.push(job);
//            condition.notify_one();
//        }
//    }
//    // 当图片遍历完时, 结束推理
////    bool running = false;
//}

void
preProcess2(ParamBase &curParam, std::vector<std::string> &imgPaths, std::vector<int> &memory, cudaStream_t &stream) {
    auto lastAddress = &imgPaths.back();
    // 若全部在gpu上完成,不再需要pin相关任何东西
    float *gpuMemoryIn = nullptr, *pinMemoryIn = nullptr;
    checkRuntime(cudaMallocHost(&pinMemoryIn, memory[0] * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
    int count = 0;
    int inputSize = memory[0] / curParam.batchSize;
    Job job;
    Out out;

    for (auto &imgPath: imgPaths) {
        // 记录队列最后一个
        if (&imgPath == lastAddress) queueFinish = true;

        // auto stream = cv::cuda::StreamAccessor::wrapStream(m_stream);
        cv::Mat mat = cv::imread(imgPath);
        float d2i[6];
        cv::Mat scaleImage = letterBox(mat, 640, 640, d2i);

//        cv::cuda::GpuMat gpuMat(scaleImage);
//        cv::cuda::cvtColor(gpuMat, gpuMat, cv::COLOR_BGR2RGB);
//        // 数值归一化
//        gpuMat.convertTo(gpuMat, CV_32FC3, 1. / 255., 0);

        // 依次存储一个batchSize中图片放射变换参数
        out.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
        if (count < curParam.batchSize) {
            BGR2RGB(scaleImage, pinMemoryIn + count * inputSize);
//            checkRuntime(cudaMemcpy(&gpuMemoryIn + count * inputSize, &gpuMat, inputSize, cudaMemcpyDeviceToDevice));
            count += 1;
        }
        // 小于一个batch,不再往下继续.
        if (count < curParam.batchSize) continue;
        // 加锁
        {
            std::unique_lock<std::mutex> l(lock_);
            condition.wait(l, [&]() {
                // false, 表示继续等待. true, 表示不等待,跳出wait
//                wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
                return qJobs.size() < 3;
            });

            //全部到gpu上,不需要这句复制
            checkRuntime(cudaMemcpy(&gpuMemoryIn, &pinMemoryIn, count * inputSize, cudaMemcpyHostToDevice));
            job.inputTensor = gpuMemoryIn;
            // 记录当前推理图片数量,正常情况count=batchSize, 但最后批次,可能会小于batchSize
            out.inferNum = count;
            // 以后取回该批次图片的结果
            job.gpuOutputPtr.reset(new std::promise<float *>());
            // 关联当前输入数据的未来输出
            out.inferResult = job.gpuOutputPtr->get_future();

            // 将存有推理数据的显存空间指针保存
            qJobs.push(job);
            qOuts.push(out);

            // 一个batch的数据已上传到推理空间
//                batchCopyOk = true;
            count = 0;
            condition.notify_one();

            // 最后图片取完,刚好够一个batch,此时不必再开辟新空间了
            if (&imgPath != lastAddress) {
                // 重新开辟一块显存, 并关联输入输出队列
                checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
                // 清空保存仿射变换参数,每个out只保留一个batch的参数
                std::vector<std::vector<float>>().swap(out.d2is);
            }
        }
    }
    // 所有处理处理完,处理不满足一个batchSize的情况
    if (count < curParam.batchSize) {
        checkRuntime(cudaMemcpy(&gpuMemoryIn, &pinMemoryIn, count * inputSize, cudaMemcpyHostToDevice));
        job.inputTensor = gpuMemoryIn;
        out.inferNum = count;
        // 以后取回该批次图片的结果
        job.gpuOutputPtr.reset(new std::promise<float *>());
        // 关联当前输入数据的未来输出
        out.inferResult = job.gpuOutputPtr->get_future();

        qJobs.push(job);
        qOuts.push(out);

        condition.notify_one();
    }
}

void postProcess(ParamBase &curParam, AlgorithmBase *curFunc, float *pinMemoryOut, std::vector<int> &memory,
                 std::vector<std::vector<std::vector<float>>> &result) {
    int singleOutputSize = memory[1] / curParam.batchSize;
    while (true) {
        if (!qOuts.empty()) {
            auto out = qOuts.front();
            cudaMemcpyAsync(pinMemoryOut, out.inferResult.get(), memory[1] * sizeof(float), cudaMemcpyDeviceToHost);
            curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, out.inferNum, result);

        }
        // 当推理结束,且推理输出队列为空,表示后处理也结束, 跳出.
        if (inferFinish && qOuts.empty()) break;

        std::this_thread::yield();
    }
}


// 通用推理接口
int trtInferProcess(ParamBase &curParam, AlgorithmBase *curFunc,
                    std::vector<std::string> &imgPaths, std::vector<std::vector<std::vector<float>>> &result) {
    //配置锁页内存,gpu显存指针
    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
    //0:当前推理模型输入tensor存储空间大小,1:当前推理输出结果存储空间大小
    std::vector<int> memory = setBatchAndInferMemory(curParam);

    // 以下,开辟内存操作不能在单独函数中完成,因为是二级指针,在当前函数中开辟内存,离开函数内存空间会消失
    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
    checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, memory[1] * sizeof(float)));
    checkRuntime(cudaMalloc(&gpuMemoryOut, memory[1] * sizeof(float)));


    // 预处理,一次处理batchSize张图片, 包括尺寸缩放,归一化,色彩转换,图片数据从内存提取到gpu
    int count = 0;
//    // 计算模型推理时,单个输入输出tensor占用空间
//    int singleInputSize = memory[0] / curParam.batchSize;
//    int singleOutputSize = memory[1] / curParam.batchSize;

    // 指定onnx中输入输出tensor名
//    curParam.context->setTensorAddress(curParam.inputName.c_str(), gpuMemoryIn);
    curParam.context->setTensorAddress(curParam.outputName.c_str(), gpuMemoryOut);

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 取得最后一个元素的地址
//    auto lastAddress = &mats.back();
    // 预处理
    std::thread t0(preProcess2, std::ref(curParam), std::ref(imgPaths), std::ref(memory), std::ref(stream));
    // 将预处理图片复制到推理空间
//    std::thread t1(uploadTensor, std::ref(curParam), std::ref(count), std::ref(*gpuMemoryIn), std::ref(singleInputSize), std::ref(stream));
    // 执行推理
    std::thread t1(trtEnqueueV3, std::ref(curParam), std::ref(count), std::ref(gpuMemoryOut), std::ref(stream));
    // 结构后处理
    std::thread t2(postProcess, std::ref(curParam), std::ref(curFunc),
                   std::ref(pinMemoryOut), std::ref(memory), std::ref(result));


    if (t0.joinable()) t0.join();
    if (t1.joinable()) t1.join();
    if (t2.joinable()) t2.join();

}


int initEngine(productParam &param, productFunc &func) {
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
        // 调用成功会返回对应模型指针对象. 失败返回nullptr
        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtYoloDetect.so"
//                "/mnt/i/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
        );
        if (!curAlg) printf("error");

        func.yoloDetect = curAlg;
        int initFlag = initCommon(param.yoloDetectParam, func.yoloDetect);
        if (0 > initFlag) {
            printf("yolo detect init failed\n");
            return -1;
        }
    }

    return 0;
}

int inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths, productResult &out) {
    // 以engine是否存在为判定,存在则执行推理
    if (nullptr != param.yoloDetectParam.engine)
        trtInferProcess(param.yoloDetectParam, func.yoloDetect, imgPaths, out.detectResult);

//    if (nullptr != conf.yoloConfig.engine)
//       trtInferProcess(conf.yoloConfig, func.yoloFace, matVector);

    return 0;
}

int getResult(productParam &param, productResult &out) {
    // 以engine是否存在为判定,存在则输出结果
//    if (nullptr != param.yoloDetectParm.engine)
//        trtInferProcess(param.yoloDetectParm, func.yoloDetect, mats, out.detectResult);
    return 0;
}
