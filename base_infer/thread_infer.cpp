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
// thread_pool经过国内某个大佬改写, 这里又改些了addTread函数
#include "../utils/thread_pool.hpp"
#include "infer.h"
#include <chrono>

class InferImpl : public Infer {
public:
    explicit InferImpl(std::vector<int> &memory);
    ~InferImpl() override;
    // 创建推理engine
    static bool getEngineContext(BaseParam &curParam);
    static std::vector<int> setBatchAndInferMemory(BaseParam &curParam);
    static unsigned long getDataLength(const InputData *data);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<int> getMemory() override {};

/* 弃用
    preProcess,postProcess空实现,具体实现由实际继承Infer.h的应用完成
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam &param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override {};
    int postProcess(BaseParam &param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};
*/
    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam *param, const cv::Mat &image, float *pinMemoryCurrentIn, const int &index) override {};

    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, int outputNums, batchBoxesType &result) override {};
    int postProcess(BaseParam *param, float *pinMemoryCurrentOut, int singleOutputSize, std::map<int, imgBoxesType> &result, const int &index) override {};

    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn) override {};
    int preProcess(BaseParam *param, const pybind11::array &image, float *pinMemoryCurrentIn, const int &index) override {};


//    具体应用调用commit方法,推理数据传入队列, 直接返回future对象. 数据依次经过trtPre,trtInfer,trtPost三个线程,结果在python代码中通过future.get()获得
    std::shared_future<batchBoxesType> commit(const InputData *data) override;

//    将待推理数据写入队列1, 会调用上述由具体应用实现的preProcess
    void trtPre(Infer *curAlg, BaseParam &curParam);
//    从队列1取数据进行推理,然后将推理结果写入队列2
    void trtInfer(BaseParam &curParam);
//    从队列2取数据,进行后处理, 会调用上述由具体应用实现的postProcess
    void trtPost(Infer *curAlg, BaseParam &curParam);

    bool startThread(Infer &curAlg, BaseParam &curParam);

private:
//   qPreLock用于预处理写入+推理时取出,qPostLock用于推理后结果写入+后处理取出, 锁与队列一一对应
    std::mutex qfJobsLock, qPreLock, qPostLock;
    std::condition_variable cv_;
//    当后处理图片数量达到totalNum时,把结果返回出来. 这个变量只使用一次. trtPre中赋值,在trtPost中使用
    unsigned long totalNum;
    float *gpuMemoryIn0 = nullptr, *gpuMemoryIn1 = nullptr, *pinMemoryIn = nullptr;
    float *gpuMemoryOut0 = nullptr, *gpuMemoryOut1 = nullptr, *pinMemoryOut = nullptr;
    float *gpuIn[2]{}, *gpuOut[2]{};

//    分别开启batch个前处理,后处理线程
    std::ThreadPool preExecutor;
    std::ThreadPool postExecutor;
//    判断当前线程是否完成
    std::vector<std::future<void>> preThreadFlags;
    std::vector<std::future<void>> postThreadFlags;

    std::vector<int> memory;
    // 读取从路径读入的图片矩阵
//    cv::Mat mat;
    std::queue<Job> qPreJobs;
// 存储每个batch的推理结果,统一后处理
    std::queue<Out> qPostJobs;
    futureJob fJob;
    std::queue<futureJob> qfJobs;
    // 记录传入的图片数量
    std::queue<int> qfJobLength;
    std::atomic<bool> preFinish{false}, inferFinish{false}, workRunning{true};
    std::shared_ptr<std::thread> preThread, inferThread, postThread;
    //创建cuda任务流,对应上述三个处理线程
    cudaStream_t preStream{}, inferStream{}, postStream{};
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Timer timer = Timer();

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

unsigned long InferImpl::getDataLength(const InputData *data) {
    unsigned long data_length;
//    if (!data->pyMats.empty())
//        data_length = data->pyMats.size();
    if (!data->mats.empty())
//    else if (!data->mats.empty())
        data_length = data->mats.size();
    else
        data_length = data->gpuMats.size();

    return data_length;
//    return 0;
}

std::shared_future<batchBoxesType> InferImpl::commit(const InputData *data) {
    // 将传入的多张或一张图片,一次性传入队列总
//    unsigned long data_length = !data->mats.empty() ? data->mats.size() : data->gpuMats.size();
//    unsigned long data_length = getDataLength(data);
//    todo C++ 调用启动mats或gpuMats, 禁用pyMats, python 引用计数在C++多线程中传递会出错
//    fJob.pyMats = data->pyMats;
//    totalNum = fJob.pyMats.size();
    fJob.mats = data->mats;
//  todo 一个get_future()对应一个get()取回结果的操作.若前一个get()未执行,后一组数据已经执行了get_future()操作, 会导致
//  todo 前一个get()操作出错. 即commit必须get()到当前数据的结果才能处理下一组数据.那么可以创建一个全局变量,统计当前传入的
//  todo 待推理图片数量,在各个线程间共用.因为在所有结果返回前不会收到下一组数据来改变这个全局变量, 所以这个变量是安全的.在commit
//  todo 中赋值, 在trtPost中使用.
    totalNum = fJob.mats.size();

//    fJob.gpuMats = data->gpuMats;

//    两种方法都可以实现初始化,make_shared更好?
    fJob.batchResult = std::make_shared<std::promise<batchBoxesType>>();
//    fJob.batchResult.reset(new std::promise<batchBoxesType>());

    // 创建share_future变量,一次性取回传入的所有图片结果, 不能直接返回xx.get_future(),会报错
    std::shared_future<batchBoxesType> future = fJob.batchResult->get_future();
    {
        std::lock_guard<std::mutex> fj(qfJobsLock);
        qfJobs.emplace(std::move(fJob));
    }
    // 通知图片预处理线程,图片已就位.
    cv_.notify_all();

    return future;
}

void InferImpl::trtPre(Infer *curAlg, BaseParam &curParam) {
    // 记录预处理总耗时
//    double totalTime;
    std::chrono::system_clock::time_point preTime;
//    auto t = timer.curTimePoint();
//    计算1个batchSize的数据所需的内存空间大小
    unsigned long mallocSize = this->memory[0] * sizeof(float);
    // count统计推理图片数量,最后一次,可能小于batchSize.imgNums统计预处理图片数量,index是gpuIn的索引,两个显存轮换使用
    int preIndex = 0, index = 0, inputSize = this->memory[0] / curParam.batchSize;
    Job job;

    while (workRunning) {
        {
            std::unique_lock<std::mutex> fj(qfJobsLock);
            // 队列不为空, 退出等待,. 当标志位workRunning为False时,说明要停止线程运行,也退出等待
            cv_.wait(fj, [&]() { return !qfJobs.empty() || !workRunning; });
//            判断退出等待原因是否是要停止线程
            if (!workRunning) break;
            fJob = qfJobs.front();
            qfJobs.pop();
        }

        // 默认推理的数量为batchSize, 只有最后一次才可能小于batchSize
        job.inferNum = curParam.batchSize;
// ---------------------------------------------------------------------------------------
//        记录待推理的最后一个元素地址
        auto lastElement = &fJob.mats.back();
        // todo 暂时,先在内存中进行图片预处理. gpuMat以后写cuda核函数处理
        for (auto &curMat: fJob.mats) {
// ---------------------------------------------------------------------------------------
            // 记录前处理耗时
//            preTime = timer.curTimePoint();
//            调用具体算法自身实现的预处理逻辑
//            curAlg->preProcess(&curParam, curMat, pinMemoryIn + preIndex * inputSize);
            preThreadFlags.emplace_back(
                    preExecutor.commit(
                            [this, &curAlg, &curParam, &inputSize, curMat, preIndex] {
//                                传入curParam的地址
                                curAlg->preProcess(&curParam, curMat, pinMemoryIn + preIndex * inputSize, preIndex);
                            })
            );
            preIndex += 1;
            // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
            if (&curMat != lastElement && preIndex < curParam.batchSize) continue;
            // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
            if (&curMat == lastElement) job.inferNum = preIndex;

//		    线程池处理方式, 所有预处理线程都完成后再继续
            for (auto &flag: preThreadFlags) {
                flag.get();
            }
////           thread_flags是存储线程池状态的vector, 每次处理完后都清理, 不然会越堆越多,内存泄露
            preThreadFlags.clear();
//            todo 必须在所有预处理线程完成后才能进行赋值, 不然会造成图片仿射变换参数与次序不一致
            // 将当前正在处理图片的变换参数加入该batch的变换vector中, 没有图像变换, 这行也不会报错, 所有模型预处理都可保留
//            在所有预处理完成后再收集仿射变换参数,确保每个索引都有值. 数量为batchSize或最后不足1个batchSize个参数
            job.d2is = curParam.preD2is;

//            totalTime += timer.timeCountS(preTime);
            {
                std::unique_lock<std::mutex> pre(qPreLock);
                // false,继续等待. true,不等待,跳出wait. 一旦进入wait,解锁. 退出wait,又加锁 最多2个batch
                cv_.wait(pre, [&]() { return qPreJobs.size() < 2; });
                //若全部在gpu上操作,不需要这句复制
                checkRuntime(cudaMemcpyAsync(gpuIn[index], pinMemoryIn, mallocSize, cudaMemcpyHostToDevice, preStream));

                // 将一个batch待推理数据的显存空间指针保存
                qPreJobs.push(job);
//               在流同步前赋值变量,会快一点点吧!
                preIndex = 0;
//               将内存地址索引指向另一块内存
                index = index >= 1 ? 0 : index + 1;
//                todo 流同步必须在锁内完成, 因为出锁后trtInfer函数会立即从qPreJobs取值,若qPreJobs仅有一个元素, 则会把这个元素取出,
//                todo 如果这时数据还没复制, 会使用上一次遗留数据推理, 若正在复制, 同时读写会造成异常.
                // 流同步,在通知队列可使用前,确保待推理数据已复制到gpu上,保证在推理时取出就能用
                checkRuntime(cudaStreamSynchronize(preStream));
            }
            cv_.notify_all();
            // 清空保存仿射变换参数,只保留当前推理batch的参数
//            todo 如果这是前面的赋值操作,似乎不必手动清理job.d2is
//            std::vector<std::vector<float >>().swap(job.d2is);
        }
    }
    // 结束预处理线程,释放资源
    preFinish = true;
    // 唤醒trt线程,告知预处理线程已结束
    cv_.notify_all();

//    logInfo("pre   use time: %.3f s", totalTime);
}

// 适用于模型推理的通用trt流程
void InferImpl::trtInfer(BaseParam &curParam) {
    // 记录推理耗时
//    double totalTime;
//    auto t = timer.curTimePoint();
    int index2 = 0;
    //    将推理后数据从从显存拷贝到内存中,计算所需内存大小,ps:其实与占用显存大小一致
    unsigned long mallocSize = this->memory[1] * sizeof(float);

//    engine输入输出节点名字, 是把model编译为onnx文件时,在onnx中手动写的输入输出节点名
    const char *inferInputName = curParam.inputName.c_str();
    const char *inferOutputName = curParam.outputName.c_str();

    Job job;
    Out trtOut;
//    第一次执行都指向0号显存空间
    curParam.context->setTensorAddress(inferInputName, gpuIn[0]);
    curParam.context->setTensorAddress(inferOutputName, gpuOut[0]);
    while (true) {
        {
            std::unique_lock<std::mutex> pre(qPreLock);
            // 队列不为空, 就说明推理空间图片已准备好,退出等待,继续推理. 当图片都处理完,并且队列为空,要退出等待,因为此时推理工作已完成
            cv_.wait(pre, [&]() { return !qPreJobs.empty() || (preFinish && qPreJobs.empty()); });
//            若图片都处理完且队列空,说明推理结束,直接退出线程
            if (preFinish && qPreJobs.empty()) break;
            job = qPreJobs.front();
//            必须在显存锁内推理, 保证推理输入数据内存不被覆盖, 如果预处理快,模型推理速度慢, 这种情况一定会出现.推理写在
//            锁外时, qPreJobs被弹出一个元素, 长度为1, 会在trtPre线程中向qPreJobs写入元素,而被写入空间就是当前准备
//            推理的数据,会造成明细的推理异常. 因此当该数据推理完成前,不准写入新数据, 即推理必须在当前互斥锁内完成.
//            printf("infer1\n");
            curParam.context->enqueueV3(inferStream);
//            printf("infer2\n");
//            推理与流同步之间弹出, 应该会节省点时间吧.
            qPreJobs.pop();
//            outTrt.inferOut = gpuOut[index2];
            trtOut.d2is = job.d2is;
            trtOut.inferNum = job.inferNum;
            cudaStreamSynchronize(inferStream);
        }
        // 消费掉一个元素,通知队列跳出等待,向qJob继续写入一个batch的待推理数据
        cv_.notify_all();
//        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // 流同步后,获取该batchSize推理结果, 然后从gpu拷贝到索引内存中
        {
            std::unique_lock<std::mutex> post(qPostLock);
            // false, 表示继续等待. true, 表示不等待,gpuOut内只有两块空间,因此队列长度只能限定为2
            cv_.wait(post, [&]() { return qPostJobs.size() < 2; });
            // todo 将engine推理好的结果从gpu转移到内存中处理, 更高效的方式是在在gpu中用cuda核函数处理,but,以后再扩展吧
//            不能让推理的输出空间被覆盖, 只有当前一个输出空间的后处理完成,并把结果存储到batchBox才能写入, 不然会出现输出结果异常
            cudaMemcpyAsync(pinMemoryOut, gpuOut[index2], mallocSize, cudaMemcpyDeviceToHost, postStream);

            qPostJobs.emplace(trtOut);
//           tensor的输入输出地址指向下一组显存空间, 在流内赋值,会快些不?
            index2 = index2 >= 1 ? 0 : index2 + 1;
            curParam.context->setTensorAddress(inferInputName, gpuIn[index2]);
            curParam.context->setTensorAddress(inferOutputName, gpuOut[index2]);

            cudaStreamSynchronize(postStream);
        }
        cv_.notify_all();
    }
    // 在post后处理线程中判断,所有推理流程是否结束,然后决定是否结束后处理线程
    inferFinish = true;
    //告知post后处理线程,推理线程已结束
    cv_.notify_all();

//    logInfo("infer use time: %.3f s", totalTime);
}

void InferImpl::trtPost(Infer *curAlg, BaseParam &curParam) {
    // 记录后处理耗时
//    double totalTime;
//    auto t = timer.curTimePoint();
    int singleOutputSize = this->memory[1] / curParam.batchSize;
//    batchBox收集每个batchSize后处理结果,然后汇总到batchBoxes中
    batchBoxesType batchBoxes, batchBox;
//    多线程后处理, 由于每张图片输出框数量不定, 不能统一初始化vector, 所以使用字典作为中转存储空间
    std::map<int, imgBoxesType> boxDict;
//    传入图片总数, 已处理图片数量
    int countPost = 0;
    Out outPost;

    while (true) {
        {
            std::unique_lock<std::mutex> post(qPostLock);
            // 队列不为空, 就说明图片已推理好,退出等待,进行后处理. 推理结束,并且队列为空,退出等待,因为此时推理工作已完成
            cv_.wait(post, [&]() { return !qPostJobs.empty() || (inferFinish && qPostJobs.empty()); });
            // 退出推理线程
            if (inferFinish && qPostJobs.empty()) break;
            outPost = qPostJobs.front();
            // 取回数据的仿射变换量,用于还原预测量在原图尺寸上的位置
            curParam.postD2is = outPost.d2is;
//          记录当前后处理图片数量, 若是单张图片,这个记录没啥用. 若是传入多个batchSize的图片,countPost会起到标识作用
            countPost += outPost.inferNum;

            curAlg->postProcess(&curParam, pinMemoryOut, singleOutputSize, outPost.inferNum, batchBox);
//          多线程后处理似乎并不快, 因为先把处理结果存储在字典boxDict中,再把结果push到vector batchBox中, 转换过程同样耗时. 如果没有
//          后处理,推理结果仅仅是二分类之类的概率值, 多线程处理反而更慢, 此时直接用上面的循环postProcess取出结果更快.
// ---------------------------------------------------------------------------------------------------------------------
//            for (int i = 0; i < outPost.inferNum; ++i) {
//                postThreadFlags.emplace_back(
//                        postExecutor.commit(
//                                [this, &curAlg, &curParam, &singleOutputSize, i, &boxDict] {
////                                为batch中每个元素进行单个处理后处理,一步获得所有处理结果
//                                    curAlg->postProcess(&curParam, pinMemoryOut, singleOutputSize, boxDict, i);
//                                })
//                );
//            }
//            //	线程池处理方式, 所有预处理线程都完成后再继续
//            for (auto &flag: postThreadFlags) {
//                flag.get();
//            }
//            postThreadFlags.clear();
////            把推理结果从字典输出到vector
//            for (int i = 0; i < outPost.inferNum; ++i) {
//                batchBox.push_back(boxDict[i]);
//            }
// ---------------------------------------------------------------------------------------------------------------------
            qPostJobs.pop();
        }
        cv_.notify_all();

        //将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());
        batchBox.clear();
        // 当commit中传入图片处理完时,通过set_value返回所有图片结果. 重新计数, 并返回下一次要输出推理结果的图片数量
        if (totalNum <= countPost) {
//            fJob.batchResult = std::make_shared<std::promise<batchBoxesType>>();
            // 输出解码后的结果,在commit中接收
            fJob.batchResult->set_value(batchBoxes);
            countPost = 0;
//            flag = true;
            batchBoxes.clear();
        }
//        totalTime += timer.timeCountS(qT1);
    }
//    logInfo("post  use time: %.3f s", totalTime);

}

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

bool InferImpl::startThread(Infer &curAlg, BaseParam &curParam) {
//	  初始化时,创建batchSize个预处理,后处理函数线程, 一步完成预处理操作
    preExecutor.addThread(static_cast<unsigned short>(curParam.batchSize));
    postExecutor.addThread(static_cast<unsigned short>(curParam.batchSize));
    try {
        preThread = std::make_shared<std::thread>(&InferImpl::trtPre, this, &curAlg, std::ref(curParam));
        inferThread = std::make_shared<std::thread>(&InferImpl::trtInfer, this, std::ref(curParam));
        postThread = std::make_shared<std::thread>(&InferImpl::trtPost, this, &curAlg, std::ref(curParam));
    } catch (std::string &error) {
        logError("thread start failure: %s !", error.c_str());
        return false;
    }

    logSuccess("thread start success !");
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

    checkRuntime(cudaMallocAsync(&gpuMemoryIn0, InMallocSize, initStream));
    checkRuntime(cudaMallocAsync(&gpuMemoryIn1, InMallocSize, initStream));

    checkRuntime(cudaMallocAsync(&gpuMemoryOut0, OutMallocSize, initStream));
    checkRuntime(cudaMallocAsync(&gpuMemoryOut1, OutMallocSize, initStream));

    checkRuntime(cudaMallocHost(&pinMemoryIn, InMallocSize));
    checkRuntime(cudaMallocHost(&pinMemoryOut, OutMallocSize));

    gpuIn[0] = gpuMemoryIn0;
    gpuIn[1] = gpuMemoryIn1;
    gpuOut[0] = gpuMemoryOut0;
    gpuOut[1] = gpuMemoryOut1;

    checkRuntime(cudaStreamDestroy(initStream));
}

InferImpl::~InferImpl() {
    logInfo("executing destructor ...");

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

    logInfo("destructor executed finished");

}

std::shared_ptr<Infer> createInfer(Infer &curAlg, BaseParam &curParam) {
//    根据外部传入gpu编号设置工作gpu
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

    // 实例化一个推理器的实现类（inferImpl），以指针形式返回
    std::shared_ptr<InferImpl> instance(new InferImpl(memory));
    //若实例化失败 或 若线程启动失败,也返回空实例. 所有的错误信息都在函数内部打印
    if (!instance || !instance->startThread(curAlg, curParam)) {
//        logError("InferImpl instance Fail");
        instance.reset();
        return nullptr;
    }

    return instance;
}