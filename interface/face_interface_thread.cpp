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
#include "../algorithm_factory/infer.h"

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
//        AlgorithmBase *curAlg = AlgorithmBase::loadDynamicLibrary(
//                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
////                "/mnt/i/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
//        );
        Infer *curAlg = InferImpl::loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
//                "/mnt/i/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
        );
        if (!curAlg) printf("error");

        auto func.yoloDetect = createInfer(param.yoloDetectParam, param.yoloDetectParam.enginePath, curAlg);
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
std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::vector<std::string> &imgPaths) {
    std::map<std::string, batchBoxesType> result;
    // 以engine是否存在为判定,存在则执行推理
    if (nullptr != param.yoloDetectParam.engine) {
        std::vector<std::string> ss;
        auto detectRes = func.yoloDetect->commit(ss);
        result["yoloDetect"] = detectRes.get();
    }
//    if (nullptr != conf.yoloConfig.engine)
//       trtInferProcess(conf.yoloConfig, func.yoloFace, matVector);

    return result;
}

int getResult(productParam &param, productResult &out) {
    // 以engine是否存在为判定,存在则输出结果
//    if (nullptr != param.yoloDetectParm.engine)
//        trtInferProcess(param.yoloDetectParm, func.yoloDetect, mats, out.detectResult);
    return 0;
}
