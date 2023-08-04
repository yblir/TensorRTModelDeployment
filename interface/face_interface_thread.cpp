//
// Created by Administrator on 2023/1/9.
//
#include <mutex>
#include <condition_variable>

#include "face_interface_thread.h"
#include "../utils/box_utils.h"
//#include "../utils/general.h"
//#include "../product/YoloDetect.h"


//int initEngine(productParam &inputParam, productFunc &func) {
int initEngine(Handle &engine, externalParam &inputParam) {
    auto *param = new productParam();
//    inputParam结构体参数从python中传入
    param->yoloDetectParam.onnxPath = inputParam.onnxPath;
    param->yoloDetectParam.gpuId = inputParam.gpuId;
    param->yoloDetectParam.batchSize = inputParam.batchSize;
    param->yoloDetectParam.inputHeight = inputParam.inputHeight;
    param->yoloDetectParam.inputWidth = inputParam.inputWidth;
    param->yoloDetectParam.inputName = inputParam.inputName;
    param->yoloDetectParam.outputName = inputParam.outputName;
    param->yoloDetectParam.iouThresh = inputParam.iouThresh;
    param->yoloDetectParam.scoreThresh = inputParam.scoreThresh;

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
    if (nullptr == param->yoloDetectParam.func) {
//        Infer *curAlg = loadDynamicLibrary(
//                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
////                "/mnt/e/GitHub/TensorRTModelDeployment/cmake-build-debug-wsl/dist/lib/libTrtYoloDetect.so"
////                "/mnt/i/GitHub/TensorRTModelDeployment/cmake-build-debug/dist/lib/libTrtYoloDetect.so"
//        );
//        if (!curAlg) printf("error");
        Infer *curAlg = new YoloDetect();
        param->yoloDetectParam.func = createInfer(param->yoloDetectParam, *curAlg);
    }
//    将收集好的参数转成void类型,返回到外部. 在python代码中, 传递到inferEngine中进行推理. 要转成void类型,因为
//    midParam在python代码中传递, ctypes只有c_void类型才能接收.
    engine = reinterpret_cast<Handle>(param);
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
int inferEngine(Handle &engine, const InputData &data) {
//   midParam从python代码中传入, 但之前在initEngine已经写入各种信息. 在此再转回productParam类型.
    auto param = reinterpret_cast<productParam *>(engine);
//    有可能多个返回结果,或多个返回依次调用,在此使用字典类型格式
    std::map<std::string, batchBoxesType> result;

    // 返回目标检测结果
    auto detectRes = param->yoloDetectParam.func->commit(data);
    result["yoloDetect"] = detectRes.get();

//    InputData data1;
//    data1.gpuMats = detectRes;
//
//    auto faceRes = func.yoloFace->commit(data1);
//    result["yoloFace"] = faceRes;


    return 0;
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

int releaseEngine(Handle &engine) {
    auto param = reinterpret_cast<productParam *>(engine);
    delete param;
}