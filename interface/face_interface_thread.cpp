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
//#include <future>

#include "../utils/general.h"
#include "face_interface_thread.h"
#include "../utils/box_utils.h"

//struct
std::mutex lock_;

std::condition_variable condition;

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

std::queue<Job> qJobs;
// 存储每个batch的推理结果,统一后处理
std::queue<Out> qOuts;
std::queue<float *> qOuts2;

std::atomic<bool> queueFinish{false};
std::atomic<bool> inferFinish{false};
Timer timer = Timer();

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

// 在gpu上开辟两块gpuMemoryIn内存,一个batch预处理数据存储一个memory中,队列长度为2,只有当其中一个memory被取走推理时,才继续预处理,写入空的memory,存入队列
//void preProcess(ParamBase &curParam, std::vector<std::string> &imgPaths, std::vector<int> &memory) {
//    // 记录预处理总耗时
//    double totalTime, totalTime2;
//    auto t = timer.curTimePoint();
//
//    auto lastAddress = &imgPaths.back();
//
//    //创建cuda任务流
//    cudaStream_t stream;
//    checkRuntime(cudaStreamCreate(&stream));
//
//    // 以下,开辟内存操作不能在单独函数中完成,因为是二级指针,在当前函数中开辟内存,离开函数内存空间会消失
//    // 若全部在gpu上完成,不再需要pin相关任何东西
//    float *gpuMemoryIn = nullptr, *pinMemoryIn = nullptr;
//    checkRuntime(cudaMallocHost(&pinMemoryIn, memory[0] * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryIn, memory[0] * sizeof(float)));
//
//    // count统计推理图片数量,正常情况下等于batchSize,但在最后一次,可能小于batchSize. imgNums统计预处理图片数量
//    int count = 0, imgNums = 0, inputSize = memory[0] / curParam.batchSize;
//    Job job;
//    Out out;
//    cv::Mat mat;
//
//    for (auto &imgPath: imgPaths) {
//        // auto stream = cv::cuda::StreamAccessor::wrapStream(m_stream);
//        mat = cv::imread(imgPath);
//        // 记录前处理耗时
//        auto preTime = timer.curTimePoint();
//        float d2i[6];
//        cv::Mat scaleImage = letterBox(mat, 640, 640, d2i);
////      -------------------------------------------------------------------------
////        cv::cuda::GpuMat gpuMat(scaleImage);
////        cv::cuda::cvtColor(gpuMat, gpuMat, cv::COLOR_BGR2RGB);
////        // 数值归一化
////        gpuMat.convertTo(gpuMat, CV_32FC3, 1. / 255., 0);
////      -------------------------------------------------------------------------
//        // 依次存储一个batchSize中图片放射变换参数
//        out.d2is.push_back({d2i[0], d2i[1], d2i[2], d2i[3], d2i[4], d2i[5]});
//        if (count < curParam.batchSize) {
//            BGR2RGB(scaleImage, pinMemoryIn + count * inputSize);
////            checkRuntime(cudaMemcpy(gpuMemoryIn + count * inputSize, &gpuMat, memory[0] * sizeof(float), cudaMemcpyDeviceToDevice));
//            count += 1;
//        }
//
//        // 小于一个batch,不再往下继续.
//        if (count < curParam.batchSize) continue;
//        // 统计图片尺寸变换等预处理耗时
//        totalTime += timer.timeCount(preTime);
//
//        //全部到gpu上,不需要这句复制
//        checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn,
//                                     memory[0] * sizeof(float), cudaMemcpyHostToDevice, stream));
////            checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn, memory[0] * sizeof(float), cudaMemcpyHostToDevice, stream));
//        job.inputTensor = gpuMemoryIn;
//        // 记录当前推理图片数量,正常情况count=batchSize, 但最后批次,可能会小于batchSize
//        out.inferNum = count;
//        // 以后取回该批次图片的结果
//        job.gpuOutputPtr.reset(new std::promise<float *>());
//        // 关联当前输入数据的未来输出
//        out.inferResult = job.gpuOutputPtr->get_future();
//
//        // 流同步,在写入队列前,确保待推理数据已复制到gpu上
//        checkRuntime(cudaStreamSynchronize(stream));
//
//        // 加锁
//        {
//            std::unique_lock<std::mutex> l(lock_);
//            // false, 表示继续等待. true, 表示不等待,跳出wait. wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
//            condition.wait(l, [&]() { return qJobs.size() < 5; });
//
//            // 将存有推理数据的显存空间指针保存
//            qJobs.push(job);
//            qOuts.push(out);
//            condition.notify_one();
//        }
//
//        imgNums += count;
//        count = 0;
//
//        // 最后图片取完,刚好够一个batch,此时不必再开辟新空间了
//        if (&imgPath != lastAddress) {
//            // 重新开辟一块显存, 并关联输入输出队列
//            checkRuntime(cudaMallocAsync(&gpuMemoryIn, memory[0] * sizeof(float), stream));
//            // 清空保存仿射变换参数,每个out只保留一个batch的参数
//            std::vector<std::vector<float>>().swap(out.d2is);
//        }
//    }
//    // 所有图片处理完,处理不满足一个batchSize的情况
//    if (count != 0 && count < curParam.batchSize) {
//        checkRuntime(cudaMemcpy(gpuMemoryIn, pinMemoryIn, count * inputSize * sizeof(float), cudaMemcpyHostToDevice));
//        job.inputTensor = gpuMemoryIn;
//        out.inferNum = count;
//        imgNums += count;
//        // 以后取回该批次图片的结果
//        job.gpuOutputPtr.reset(new std::promise<float *>());
//        // 关联当前输入数据的未来输出
//        out.inferResult = job.gpuOutputPtr->get_future();
//        {
//            std::lock_guard<std::mutex> l(lock_);
//            qJobs.push(job);
//            qOuts.push(out);
//            condition.notify_all();
//        }
//    }
//    queueFinish = true;
//
//    // 线程结束时,仅是否pinMemoryIn, gpuMemoryIn存有推理数据,在推理线程中释放
//    if (pinMemoryIn) checkRuntime(cudaFreeHost(pinMemoryIn));
//    totalTime2 = timer.timeCount(t);
//    printf("pre   use time: %.2f ms, thread use time: %.2f ms, pre img num = %d\n", totalTime, totalTime2, imgNums);
//    checkRuntime(cudaStreamDestroy(stream));
//}

void preProcess(ParamBase &curParam, std::vector<std::string> &imgPaths, std::vector<int> &memory) {
    // 记录预处理总耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();

    auto lastAddress = &imgPaths.back();
    unsigned long mallocSize = memory[0] * sizeof(float);
    //创建cuda任务流
    cudaStream_t stream;
    checkRuntime(cudaStreamCreate(&stream));

    // 以下,开辟内存操作不能在单独函数中完成,因为是二级指针,在当前函数中开辟内存,离开函数内存空间会消失
    // 若全部在gpu上完成,不再需要pin相关任何东西
    float *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *gpuMemoryIn2 = nullptr, *pinMemoryIn = nullptr;
    checkRuntime(cudaMallocHost(&pinMemoryIn, mallocSize));
    checkRuntime(cudaMallocAsync(&gpuMemoryIn0, mallocSize, stream));
    checkRuntime(cudaMallocAsync(&gpuMemoryIn1, mallocSize, stream));
    checkRuntime(cudaMallocAsync(&gpuMemoryIn2, mallocSize, stream));
    float *gpuIn[] = {gpuMemoryIn0, gpuMemoryIn1, gpuMemoryIn2};

    // count统计推理图片数量,正常情况下等于batchSize,但在最后一次,可能小于batchSize.imgNums统计预处理图片数量,index是gpuIn的索引,三个显存轮换使用
    int count = 0, imgNums = 0, index = 0, inputSize = memory[0] / curParam.batchSize;
    Job job;
    Out out;
    cv::Mat mat;

    for (auto &imgPath: imgPaths) {
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
        out.inferNum = count;
        // 以后取回该批次图片的结果
        job.gpuOutputPtr.reset(new std::promise<float *>());
        // 关联当前输入数据的未来输出
        out.inferResult = job.gpuOutputPtr->get_future();

        // 流同步,在写入队列前,确保待推理数据已复制到gpu上
        checkRuntime(cudaStreamSynchronize(stream));

        // 加锁
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait. wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
            condition.wait(l, [&]() { return qJobs.size() < 3; });

            // 将存有推理数据的显存空间指针保存
            qJobs.push(job);
            qOuts.push(out);
            condition.notify_one();
        }

        imgNums += count;
        count = 0;
        // 轮换到下一个显存空间
        index = index >= 2 ? 0 : index + 1;
        // 清空保存仿射变换参数,每个out只保留一个batch的参数
        if (&imgPath != lastAddress) std::vector<std::vector<float>>().swap(out.d2is);

    }
    // 所有图片处理完,处理不满足一个batchSize的情况
    if (count != 0 && count < curParam.batchSize) {
        checkRuntime(cudaMemcpy(gpuIn[index], pinMemoryIn, count * inputSize * sizeof(float), cudaMemcpyHostToDevice));
        job.inputTensor = gpuIn[index];
        out.inferNum = count;
        imgNums += count;
        // 以后取回该批次图片的结果
        job.gpuOutputPtr.reset(new std::promise<float *>());
        // 关联当前输入数据的未来输出
        out.inferResult = job.gpuOutputPtr->get_future();
        // 加锁
        {
            std::unique_lock<std::mutex> l(lock_);
            condition.wait(l, [&]() { return qJobs.size() < 3; });
            // 将存有推理数据的显存空间指针保存
            qJobs.push(job);
            qOuts.push(out);
            condition.notify_one();
        }
    }
    queueFinish = true;

    // 线程结束时,仅是否pinMemoryIn, gpuMemoryIn存有推理数据,在推理线程中释放
//    for (auto &item: gpuIn) checkRuntime(cudaFreeAsync(item, stream));
    checkRuntime(cudaFreeHost(pinMemoryIn));
    totalTime2 = timer.timeCount(t);
    printf("pre   use time: %.2f ms, thread use time: %.2f ms, pre img num = %d\n", totalTime, totalTime2, imgNums);
    checkRuntime(cudaStreamDestroy(stream));
}

//// 适用于模型推理的通用trt流程
//int trtEnqueueV3(ParamBase &curParam, std::vector<int> &memory) {
//    // 记录推理耗时
//    double totalTime, totalTime2;
//    auto t = timer.curTimePoint();
//    unsigned long mallocSize = memory[1] * sizeof(float);
//    //创建cuda任务流
//    cudaStream_t stream;
//    checkRuntime(cudaStreamCreate(&stream));
//
//    float *gpuMemoryOut = nullptr;
//    checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
//    curParam.context->setTensorAddress(curParam.outputName.c_str(), gpuMemoryOut);
//    Job job;
//    while (true) {
//        {
//            std::unique_lock<std::mutex> l(lock_);
//            // false, 表示继续等待. true, 表示不等待,跳出wait
//            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
//            condition.wait(l, [&]() { return !qJobs.empty(); });
//            job = qJobs.front();
//            qJobs.pop();
//            // 消费掉一个元素,通知队列跳出等待,继续生产
//            condition.notify_one();
//        }
//        curParam.context->setTensorAddress(curParam.inputName.c_str(), job.inputTensor);
//
//        // 记录推理耗时
//        auto qT1 = timer.curTimePoint();
//        // 执行异步推理
//        curParam.context->enqueueV3(stream);
//        // 流同步
//        cudaStreamSynchronize(stream);
//
//        totalTime += timer.timeCount(qT1);
//        // 流同步后,获取该batchSize推理结果
//        job.gpuOutputPtr->set_value(gpuMemoryOut);
//        // 再判断一次推理数据队列,若图片都处理完且队列空,说明推理结束,不必再开辟新空间了,直接退出线程
//        if (queueFinish && qJobs.empty()) break;
//
//        // 重新开辟推理输出空间,保证每个推理输出空间都不同,避免多线程推理时的结果覆盖
//        checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
//    }
//    inferFinish = true;
//    totalTime2 = timer.timeCount(t);
//    printf("infer use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
//    checkRuntime(cudaStreamDestroy(stream));
//}

// 适用于模型推理的通用trt流程
int trtEnqueueV3(ParamBase &curParam, std::vector<int> &memory) {
    // 记录推理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
    unsigned long mallocSize = memory[1] * sizeof(float);
    //创建cuda任务流
    cudaStream_t stream;
    checkRuntime(cudaStreamCreate(&stream));

    float *gpuMemoryOut = nullptr;
    checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
    curParam.context->setTensorAddress(curParam.outputName.c_str(), gpuMemoryOut);
    Job job;
    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            condition.wait(l, [&]() { return !qJobs.empty(); });
            job = qJobs.front();
            qJobs.pop();
            // 消费掉一个元素,通知队列跳出等待,继续生产
            condition.notify_one();
        }
        curParam.context->setTensorAddress(curParam.inputName.c_str(), job.inputTensor);

        // 记录推理耗时
        auto qT1 = timer.curTimePoint();
        // 执行异步推理
        curParam.context->enqueueV3(stream);
        // 流同步
        cudaStreamSynchronize(stream);

        totalTime += timer.timeCount(qT1);
        // 流同步后,获取该batchSize推理结果
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait. wait流程: 一旦进入wait,解锁. 退出wait,又加锁 最多3个batch
            condition.wait(l, [&]() { return qOuts2.size() < 3; });
            qOuts2.push(gpuMemoryOut);
            condition.notify_one();
        }

        // 再判断一次推理数据队列,若图片都处理完且队列空,说明推理结束,不必再开辟新空间了,直接退出线程
        if (queueFinish && qJobs.empty()) break;

        // 重新开辟推理输出空间,保证每个推理输出空间都不同,避免多线程推理时的结果覆盖
        checkRuntime(cudaMallocAsync(&gpuMemoryOut, mallocSize, stream));
    }
    inferFinish = true;
    totalTime2 = timer.timeCount(t);
    printf("infer use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
    checkRuntime(cudaStreamDestroy(stream));
}

void postProcess(ParamBase &curParam, AlgorithmBase *curFunc, std::vector<int> &memory,
                 std::shared_future<std::vector<std::vector<std::vector<float>>>> &result) {
    // 记录后处理耗时
    double totalTime, totalTime2;
    auto t = timer.curTimePoint();
//    std::shared_future<std::vector<std::vector<std::vector<float>>>> result;

    int mallocSize = memory[1] * sizeof(float), singleOutputSize = memory[1] / curParam.batchSize;
    float *pinMemoryOut = nullptr;
    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
    checkRuntime(cudaMallocHost(&pinMemoryOut, mallocSize));

    float *out;
    while (true) {
        {
            std::unique_lock<std::mutex> l(lock_);
            // false, 表示继续等待. true, 表示不等待,跳出wait
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            condition.wait(l, [&]() { return !qOuts2.empty() || (inferFinish && qOuts2.empty()); });
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

//void postProcess(ParamBase &curParam, AlgorithmBase *curFunc,
//                 std::vector<int> &memory, std::vector<std::vector<std::vector<float>>> &result) {
//    // 记录后处理耗时
//    double totalTime, totalTime2;
//    auto t = timer.curTimePoint();
//
//    //创建cuda任务流
//    cudaStream_t stream;
//    checkRuntime(cudaStreamCreate(&stream));
//
//    int mallocSize = memory[1] * sizeof(float), singleOutputSize = memory[1] / curParam.batchSize;
//    float *pinMemoryOut = nullptr;
//    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, mallocSize));
//
//    Out out;
//    while (true) {
//        if (!qOuts.empty()) {
//            {
//                std::lock_guard<std::mutex> l(lock_);
//                out = qOuts.front();
//                // 把当前取出的元素删掉
//                qOuts.pop();
//            }
//            float *futureResult = out.inferResult.get();
//            // 转移到内存中处理
//            cudaMemcpy(pinMemoryOut, futureResult, mallocSize, cudaMemcpyDeviceToHost);
//            curParam.d2is = out.d2is;
//            auto qT1 = timer.curTimePoint();
//            curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, out.inferNum, result);
//            totalTime += timer.timeCount(qT1);
//
//            // 清理已用完的gpuMemoryOut
//            checkRuntime(cudaFreeAsync(futureResult, stream));
//            // 退出推理线程
//            if (inferFinish && qOuts.empty()) break;
//        } else {
//            std::unique_lock<std::mutex> l(lock_);
//            // false, 表示继续等待. true, 表示不等待,跳出wait
//            // 队列不为空, 说明结果队列可以取推理数据了
//            condition.wait(l, [&]() { return !qOuts.empty(); });
//        }
//    }
//    // 后处理全部完成,释放存储推理结果的pinMemoryOut
//    if (pinMemoryOut) checkRuntime(cudaFreeHost(pinMemoryOut));
//    totalTime2 = timer.timeCount(t);
//    printf("post  use time: %.2f ms, thread use time: %.2f ms\n", totalTime, totalTime2);
//    checkRuntime(cudaStreamDestroy(stream));
//}

// 通用推理接口
int trtInferProcess(ParamBase &curParam, AlgorithmBase *curFunc,
                    std::vector<std::string> &imgPaths, std::vector<std::vector<std::vector<float>>> &result) {

    //0:当前推理模型输入tensor存储空间大小,1:当前推理输出结果存储空间大小
    std::vector<int> memory = setBatchAndInferMemory(curParam);

    // 预处理
    std::thread t0(preProcess, std::ref(curParam), std::ref(imgPaths), std::ref(memory));
    // 执行推理
    std::thread t1(trtEnqueueV3, std::ref(curParam), std::ref(memory));
//    std::thread t12(trtEnqueueV3, std::ref(curParam), std::ref(memory));
    // 结果后处理
    std::thread t2(postProcess, std::ref(curParam), std::ref(curFunc), std::ref(memory), std::ref(result));

    if (t0.joinable()) t0.join();
    if (t1.joinable()) t1.join();
//    if (t12.joinable()) t12.join();
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
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
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

//// imgPaths图片数量为多少, 就一次性返回多少个输出结果.分批传入图片的逻辑由调用程序控制
//int inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths, productResult &out) {
//    // 以engine是否存在为判定,存在则执行推理
//    if (nullptr != param.yoloDetectParam.engine)
//        /*
//         *  for(ff 图片)
//                vector(存储有一个batch的结果) = inferengine( 图片路径)
//         * */
//        trtInferProcess(param.yoloDetectParam, func.yoloDetect, imgPaths, out.detectResult);
//
////    if (nullptr != conf.yoloConfig.engine)
////       trtInferProcess(conf.yoloConfig, func.yoloFace, matVector);
//
//    return 0;
//}

// imgPaths图片数量为多少, 就一次性返回多少个输出结果.分批传入图片的逻辑由调用程序控制
 std::map<std::string,std::vector<std::vector<float>>> inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths, productResult &out) {
    // 以engine是否存在为判定,存在则执行推理
    if (nullptr != param.yoloDetectParam.engine)
        /*
         *  for(ff 图片)
                vector(存储有一个batch的结果) = inferengine( 图片路径)
         * */
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
