//
// Created by Administrator on 2023/1/9.
//
#include <mutex>
#include <condition_variable>

#include "face_interface_thread.h"
#include "../utils/box_utils.h"
//#include "../utils/general.h"



//int initEngine(productParam &param, productFunc &func) {
int initEngine(productParam &param) {
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
    if (nullptr == param.yoloDetectParam.func) {
        Infer *curAlg = loadDynamicLibrary(
                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
//                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtYoloDetect.so"
//                "/mnt/i/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
        );
        if (!curAlg) printf("error");
        param.yoloDetectParam.func= createInfer(param.yoloDetectParam, *curAlg);
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
// productParam &param, productFunc &func, std::vector<std::string> &imgPaths
// imgPaths图片数量为多少, 就一次性返回多少个输出结果.分批传入图片的逻辑由调用程序控制
//std::map<std::string, batchBoxesType> inferEngine(productFunc &func, const InputData &data) {
std::map<std::string, batchBoxesType> inferEngine(productParam &param,const InputData &data) {
//    有可能多个返回结果,或多个返回依次调用,在此使用字典类型格式
    std::map<std::string, batchBoxesType> result;

    // 返回目标检测结果
    auto detectRes = param.yoloDetectParam.func->commit(data);
    result["yoloDetect"] = detectRes.get();

//    InputData data1;
//    data1.gpuMats = detectRes;
//
//    auto faceRes = func.yoloFace->commit(data1);
//    result["yoloFace"] = faceRes;


    return result;
}

//std::map<std::string, batchBoxesType> inferEngine(productParam &param, productFunc &func, std::vector<cv::Mat> &mats){
//
//}
//
//int getResult(productParam &param, productResult &out) {
//    // 以engine是否存在为判定,存在则输出结果
////    if (nullptr != param.yoloDetectParm.engine)
////        trtInferProcess(param.yoloDetectParm, func.yoloDetect, mats, out.detectResult);
//    return 0;
//}

//int releaseEngine() {
//    ~productFunc();
//}