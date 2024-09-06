//
// Created by Administrator on 2023/1/9.
//

#include "thread_interface.h"


int Engine::initEngine(const ManualParam &inputParam) {
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

    curAlgParam->mode = inputParam.fp32 ? Mode::FP32 : Mode::FP16;

    curAlgParam->onnxPath = inputParam.onnxPath;
    curAlgParam->enginePath = inputParam.enginePath;

    curAlgParam->inputName = inputParam.inputName;
    curAlgParam->outputName = inputParam.outputName;

    // 初始化tensorrt engine
    curAlgParam->trt = createInfer(*curAlg, *curAlgParam);
    if (nullptr == curAlgParam->trt) {
        logError("create tensorrt engine failed");
        return -1;
    }

    // 对仿射参数的初始化,用于占位, 后续使用多线程预处理, 图片参数根据传入次序在指定位置插入
    for (int i = 0; i < curAlgParam->batchSize; ++i) {
        curAlgParam->preD2is.push_back({0., 0., 0., 0., 0., 0.});
    }

    return 0;
}

#ifdef PYBIND11
futureBoxes Engine::inferEngine(const pybind11::array &img) {
//    pybind11::gil_scoped_release release;
//  todo 为避免引用计数引起的段错误, 单张推理也直接转为cv::Mat格式
//  每次调用前,清理上一次遗留的数据.
    data->mats.clear();
//    array转成cv::Mat格式
    cv::Mat mat(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));

//  emplace_back效率比data->mats= std::vector<cv::Mat> {mat}更高一些.
    data->mats.emplace_back(mat);
//    data->mats= std::vector<cv::Mat> {mat};

    return curAlgParam->trt->commit(data);
}

futureBoxes Engine::inferEngine(const std::vector<pybind11::array> &imgs) {
//    todo 不能直接以pybind11::array传入commit中,必须在此处转为cv::Mat格式. 因为imgs是从python中传入的
//    todo list,元素是python, 在trtPre,preprocess多个线程来回传递会造成python引用计数异常, 产生segment error段错误.
//    将读入的所有图片路径转为cv::Mat格式
    std::vector<cv::Mat> mats;
    for (auto &img: imgs)
        mats.emplace_back(img.shape(0), img.shape(1), CV_8UC3, (unsigned char *) img.data(0));

    data->mats = mats;

    return curAlgParam->trt->commit(data);

}
#endif

futureBoxes Engine::inferEngine(const cv::Mat &mat) const {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
//    futureBoxes result;
//  每次调用前,清理上一次遗留的数据.
    data->mats.clear();
    data->mats.emplace_back(mat);
//    返回目标检测结果
//    auto futureResult = curAlgParam->func->commit(data);
    return curAlgParam->trt->commit(data);
}

futureBoxes Engine::inferEngine(const std::vector<cv::Mat> &mats) const {
//    有可能多个返回结果, 或多个返回依次调用, 在此使用字典类型格式
//    futureBoxes result;
    data->mats = mats;
//    返回目标检测结果
//    auto futureResult = curAlgParam->func->commit(data);
    return curAlgParam->trt->commit(data);
}

void Engine::releaseEngine() const {
    delete curAlg;
    delete curAlgParam;
    delete data;
    logSuccess("Release engine success");
}

#ifdef PYBIND11
PYBIND11_MODULE(deployment, m) {
//    配置手动输入参数
    pybind11::class_<ManualParam>(m, "ManualParam")
            .def(pybind11::init<>())
//            .def_readwrite("fp16", &ManualParam::fp16)
            .def_readwrite("fp32", &ManualParam::fp32)
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

            .def("inferEngine",
                pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine),
                 pybind11::arg("image"),
                 pybind11::call_guard<pybind11::gil_scoped_release>()
            )
            .def("inferEngine",
                pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine),
                 pybind11::arg("images"),
                 pybind11::call_guard<pybind11::gil_scoped_release>()
            )

            .def("releaseEngine", &Engine::releaseEngine);
}
#endif