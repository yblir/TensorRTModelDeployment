//
// Created by Administrator on 2023/1/9.
//
#include <mutex>
#include <condition_variable>
#include "../product/product.h"
#include "face_interface_thread_pybind11.h"


int Engine::initEngine(ManualParam &inputParam) {
    param = new productParam;
//    inputParam结构体参数从python中传入
    printf("0000\n");
    std::cout << inputParam.iouThresh << std::endl;
    std::cout << inputParam.scoreThresh << std::endl;

    printf("0000\n");
    float a = inputParam.iouThresh;
    std::cout << a << " = " << a << std::endl;
    float b = inputParam.scoreThresh;
    std::cout << b << " = " << b << std::endl;
    printf("-------------\n");

    param->yoloDetectParam.iouThresh = a;
    printf("0000\n");
    param->yoloDetectParam.scoreThresh = b;
    printf("1111\n");
    std::string s = inputParam.onnxPath;
    std::cout << s << "=" << s << std::endl;
    printf("=============\n");
    param->yoloDetectParam.onnxPath = inputParam.onnxPath;
//    strcpy(param->yoloDetectParam.onnxPath, inputParam.onnxPath);
    printf("22222\n");
    param->yoloDetectParam.gpuId = inputParam.gpuId;
    param->yoloDetectParam.batchSize = inputParam.batchSize;
    printf("33333\n");
    param->yoloDetectParam.inputHeight = inputParam.inputHeight;
    param->yoloDetectParam.inputWidth = inputParam.inputWidth;
    param->yoloDetectParam.inputName = inputParam.inputName;
    printf("4444\n");
    param->yoloDetectParam.outputName = inputParam.outputName;
    printf("5555\n");
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
        Infer *curAlg = new YoloDetect();
        param->yoloDetectParam.func = createInfer(param->yoloDetectParam, *curAlg);
    }
//    将收集好的参数转成void类型,返回到外部. 在python代码中, 传递到inferEngine中进行推理. 要转成void类型,因为
//    midParam在python代码中传递, ctypes只有c_void类型才能接收.
    return 0;
}

std::map<std::string, futureBoxes> Engine::inferEngine(const std::string &imgPath) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
    std::map<std::string, futureBoxes> result;
    std::vector<cv::Mat> mats;
    mats.emplace_back(cv::imread(imgPath));

    result = inferEngine(mats);
    return result;
}

std::map<std::string, futureBoxes> Engine::inferEngine(const std::vector<std::string> &imgPaths) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
    std::map<std::string, futureBoxes> result;
//    将读入的所有图片路径转为cv::Mat格式
    std::vector<cv::Mat> mats;

    for (auto &imgPath: imgPaths) {
        mats.emplace_back(cv::imread(imgPath));
    }

    result = inferEngine(mats);

    return result;
}

std::map<std::string, futureBoxes> Engine::inferEngine(const pybind11::array &img) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
    std::map<std::string, futureBoxes> result;
    std::vector<cv::Mat> mats;

//    array转成cv::Mat格式
    cv::Mat mat(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));
    mats.emplace_back(mat);

    result = inferEngine(mats);

    return result;
}

std::map<std::string, futureBoxes> Engine::inferEngine(const std::vector<pybind11::array> &imgs) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
    std::map<std::string, futureBoxes> result;
//    将读入的所有图片路径转为cv::Mat格式
    std::vector<cv::Mat> mats;
    for (auto &img: imgs)
        mats.emplace_back(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));

    result = inferEngine(mats);
    return result;
}

std::map<std::string, futureBoxes> Engine::inferEngine(const std::vector<cv::Mat> &mats) {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
    std::map<std::string, futureBoxes> result;
    data->mats = mats;
//    返回目标检测结果
    auto futureResult = param->yoloDetectParam.func->commit(data);
    // 对返回的结果进行.get()操作,可获得结果
    result["yolo"] = futureResult;

//    InputData data1;
//    data1.gpuMats = detectRes;
//
//    auto faceRes = func.yoloFace->commit(data1);
//    result["yoloFace"] = faceRes;
    return result;
}

int Engine::releaseEngine() {
    delete param;
    delete data;
}

PYBIND11_MODULE(deployment, m) {
    //    配置手动输入参数
    pybind11::class_<ManualParam>(m, "ManualParam")
            .def(pybind11::init<>())
            .def_readwrite("gpuId", &ManualParam::gpuId)
            .def_readwrite("batchSize", &ManualParam::batchSize)

            .def_readwrite("scoreThresh", &ManualParam::scoreThresh)
            .def_readwrite("iouThresh", &ManualParam::iouThresh)
            .def_readwrite("classNums", &ManualParam::classNums)

            .def_readwrite("inputHeight", &ManualParam::inputHeight)
            .def_readwrite("inputWidth", &ManualParam::inputWidth)

            .def_readwrite("onnxPath", &ManualParam::onnxPath)
            .def_readwrite("inputName", &ManualParam::inputName)
            .def_readwrite("outputName", &ManualParam::outputName);

//    暴露的推理接口
    pybind11::class_<Engine>(m, "Engine")
            .def(pybind11::init<>())
            .def("initEngine", &Engine::initEngine)

            .def("inferEngine", pybind11::overload_cast<const std::string &>(&Engine::inferEngine))
            .def("inferEngine", pybind11::overload_cast<const std::vector<std::string> &>(&Engine::inferEngine))
            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine))
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine))

            .def("releaseEngine", &Engine::releaseEngine);
}