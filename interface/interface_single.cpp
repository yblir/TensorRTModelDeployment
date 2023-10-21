//
// Created by Administrator on 2023/1/9.
//

#include "interface_single.h"


int Engine::initEngine(ManualParam &inputParam) {
    param = new productParam;
    data = new InputData;

//    inputParam结构体参数从python中传入
    param->yoloDetectParam.gpuId = inputParam.gpuId;
    param->yoloDetectParam.batchSize = inputParam.batchSize;

    param->yoloDetectParam.inputHeight = inputParam.inputHeight;
    param->yoloDetectParam.inputWidth = inputParam.inputWidth;

    param->yoloDetectParam.mode = inputParam.fp16 ? Mode::FP16 : Mode::FP32;

    param->yoloDetectParam.onnxPath = inputParam.onnxPath;
    param->yoloDetectParam.enginePath = inputParam.enginePath;

    param->yoloDetectParam.inputName = inputParam.inputName;
    param->yoloDetectParam.outputName = inputParam.outputName;

    Infer *curAlg = new YoloDetect();
    param->yoloDetectParam.func = createInfer(param->yoloDetectParam, *curAlg);
    if (nullptr == param->yoloDetectParam.func) {
        return -1;
    }

    return 0;
}


batchBoxesType Engine::inferEngine(const pybind11::array &image) {
    pybind11::gil_scoped_release release;
//    cv::Mat mat(image.shape(1), image.shape(2), CV_8UC3, (unsigned char *) image.data(0));
    return param->yoloDetectParam.func->commit(&param->yoloDetectParam, data);
//    return inferEngine(mats);
}

batchBoxesType Engine::inferEngine(const std::vector<pybind11::array> &images) {
    pybind11::gil_scoped_release release;
//    cv::Mat mat(image.shape(1), image.shape(2), CV_8UC3, (unsigned char *) image.data(0));
    return param->yoloDetectParam.func->commit(&param->yoloDetectParam, data);
//    return inferEngine(mats);
}
//futureBoxes Engine::inferEngine(const std::vector<pybind11::array> &images) {
////    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
////    futureBoxes result;
////    将读入的所有图片路径转为cv::Mat格式
////    logInfo("arrays inferEngine ...");
//    std::vector<cv::Mat> mats;
////    for (auto &image: images) {
//////        std::cout << "image.shape=" << image.shape() << std::endl;
////        mats.emplace_back(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));
////    }
//    data->mats = images;
//    return param->yoloDetectParam.func->commit(data);
//
////    return inferEngine(mats);
//}

//futureBoxes Engine::inferEngine(const std::vector<cv::Mat> &mats) {
////    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
////    futureBoxes result;
////    data->mats = mats;
////    返回目标检测结果
////    auto futureResult = param->yoloDetectParam.func->commit(data);
//
//    return param->yoloDetectParam.func->commit(data);
//}

int Engine::releaseEngine() {
    delete param;
    delete data;
    logSuccess("Release engine success");
}
//
PYBIND11_MODULE(deployment, m) {
//    配置手动输入参数
    pybind11::class_<ManualParam>(m, "ManualParam")
            .def(pybind11::init<>())
            .def_readwrite("fp16", &ManualParam::fp16)
            .def_readwrite("gpuId", &ManualParam::gpuId)
            .def_readwrite("batchSize", &ManualParam::batchSize)

//            .def_readwrite("scoreThresh", &ManualParam::scoreThresh)
//            .def_readwrite("iouThresh", &ManualParam::iouThresh)
//            .def_readwrite("classNums", &ManualParam::classNums)

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
            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine), pybind11::arg("image"))
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine), pybind11::arg("images"))

            .def("releaseEngine", &Engine::releaseEngine);
}