//
// Created by Administrator on 2023/1/9.
//
//#include <mutex>
//#include <condition_variable>
//#include "../product/product.h"
#include "thread_interface.h"


int Engine::initEngine(ManualParam &inputParam) {
//    param = new productParam;
    data = new InputData;
    curAlg = new YoloDetect();
    curAlgParam = new YoloDetectParam;

    if (inputParam.batchSize > curAlgParam->maxBatch) {
        logError("input batchSize more than maxBatch of built engine,the maxBatch is 16");
        return -1;
    }

//    inputParam结构体参数从python中传入
    curAlgParam->classNums = inputParam.classNums;
    curAlgParam->scoreThresh = inputParam.scoreThresh;
    curAlgParam->iouThresh = inputParam.iouThresh;

    curAlgParam->gpuId = inputParam.gpuId;
    curAlgParam->batchSize = inputParam.batchSize;

    curAlgParam->inputHeight = inputParam.inputHeight;
    curAlgParam->inputWidth = inputParam.inputWidth;

    curAlgParam->mode = inputParam.fp16 ? Mode::FP16 : Mode::FP32;

    curAlgParam->onnxPath = inputParam.onnxPath;
    curAlgParam->enginePath = inputParam.enginePath;

    curAlgParam->inputName = inputParam.inputName;
    curAlgParam->outputName = inputParam.outputName;

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

    curAlgParam->trt = createInfer(*curAlg, *curAlgParam);
    if (nullptr == curAlgParam->trt) {
        return -1;
    }
    return 0;
}

//futureBoxes Engine::inferEngine(const std::string &imgPath) {
////    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
////    futureBoxes result;
//    std::vector<cv::Mat> mats;
//    mats.emplace_back(cv::imread(imgPath));
//
////    result = inferEngine(mats);
//    return inferEngine(mats);
//}
//
//futureBoxes Engine::inferEngine(const std::vector<std::string> &imgPaths) {
////    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
////    futureBoxes result;
////    将读入的所有图片路径转为cv::Mat格式
//    std::vector<cv::Mat> mats;
//
//    for (auto &imgPath: imgPaths) {
//        mats.emplace_back(cv::imread(imgPath));
//    }
//
////    result = inferEngine(mats);
//
//    return inferEngine(mats);
//}

futureBoxes Engine::inferEngine(const pybind11::array &img) {
//    pybind11::gil_scoped_release release;
//    std::vector<cv::Mat> mats;
//    array转成cv::Mat格式
//    cv::Mat mat(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));
//    mats.emplace_back(mat);

    data->pyMats.push_back(img);
    return curAlgParam->trt->commit(data);
}

futureBoxes Engine::inferEngine(const std::vector<pybind11::array> &imgs) {
//    pybind11::gil_scoped_release release;
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
//    futureBoxes result;
//    将读入的所有图片路径转为cv::Mat格式
//    std::vector<cv::Mat> mats;
//    for (auto &img: imgs)
//        mats.emplace_back(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));
    data->pyMats = imgs;
//    result = inferEngine(mats);
    return curAlgParam->trt->commit(data);
}

futureBoxes Engine::inferEngine(const std::vector<cv::Mat> &mats) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
//    futureBoxes result;
    data->mats = mats;
//    返回目标检测结果
//    auto futureResult = curAlgParam->func->commit(data);
    return curAlgParam->trt->commit(data);
}

int Engine::releaseEngine() {
    delete curAlg;
    delete curAlgParam;
    delete data;
    logSuccess("Release engine success");
}

PYBIND11_MODULE(deployment, m) {
//    配置手动输入参数
    pybind11::class_<ManualParam>(m, "ManualParam")
            .def(pybind11::init<>())
            .def_readwrite("fp16", &ManualParam::fp16)
            .def_readwrite("gpuId", &ManualParam::gpuId)
            .def_readwrite("batchSize", &ManualParam::batchSize)

            .def_readwrite("scoreThresh", &ManualParam::scoreThresh)
            .def_readwrite("iouThresh", &ManualParam::iouThresh)
            .def_readwrite("classNums", &ManualParam::classNums)

            .def_readwrite("inputHeight", &ManualParam::inputHeight)
            .def_readwrite("inputWidth", &ManualParam::inputWidth)

            .def_readwrite("onnxPath", &ManualParam::onnxPath)
            .def_readwrite("enginePath", &ManualParam::enginePath)

            .def_readwrite("inputName", &ManualParam::inputName)
            .def_readwrite("outputName", &ManualParam::outputName);

//    注册返回到python中的future数据类型,并定义get方法. 不然inferEngine返回的结果类型会出错
    pybind11::class_<futureBoxes>(m, "SharedFutureObject")
            .def("get", &futureBoxes::get);

//    暴露的推理接口
    pybind11::class_<Engine>(m, "Engine")
            .def(pybind11::init<>())
            .def("initEngine", &Engine::initEngine)

//            .def("inferEngine", pybind11::overload_cast<const std::string &>(&Engine::inferEngine))
//            .def("inferEngine", pybind11::overload_cast<const std::vector<std::string> &>(&Engine::inferEngine))

            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine),
                 pybind11::arg("image"), pybind11::call_guard<pybind11::gil_scoped_release>()
            )
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine),
                 pybind11::arg("images"), pybind11::call_guard<pybind11::gil_scoped_release>()
            )

            .def("releaseEngine", &Engine::releaseEngine);
}