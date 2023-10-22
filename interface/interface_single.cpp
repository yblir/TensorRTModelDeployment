//
// Created by Administrator on 2023/1/9.
//
#include <cuda_runtime.h>
#include "interface_single.h"


int Engine::initEngine(ManualParam &inputParam) {
    param = new productParam;
//    data = new InputData;
    if (inputParam.batchSize > param->yoloDetectParam.maxBatch) {
        logError("input batchSize more than maxBatch of built engine");
        return -1;
    }

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

    curAlg = new YoloDetect();
    param->yoloDetectParam.func = createInfer(param->yoloDetectParam, *curAlg);
    if (nullptr == param->yoloDetectParam.func) {
        return -1;
    }

//    // 更新trt context内容, 并计算trt推理需要的输入输出内存大小
    auto memory = param->yoloDetectParam.func->getMemory();
    trtInSize = memory[0] * sizeof(float);
    trtOutSize = memory[1] * sizeof(float);
    singleInputSize = memory[0] / param->yoloDetectParam.batchSize;
    singleOutputSize = memory[1] / param->yoloDetectParam.batchSize;

    checkRuntime(cudaMallocAsync(&gpuIn, trtInSize, commitStream));
    checkRuntime(cudaMallocAsync(&gpuOut, trtOutSize, commitStream));
    checkRuntime(cudaMallocHost(&pinMemoryIn, trtInSize));
    checkRuntime(cudaMallocHost(&pinMemoryOut, trtOutSize));

    param->yoloDetectParam.context->setTensorAddress(param->yoloDetectParam.inputName.c_str(), gpuIn);
    param->yoloDetectParam.context->setTensorAddress(param->yoloDetectParam.outputName.c_str(), gpuOut);

    cudaStreamSynchronize(commitStream);

    return 0;
}

// todo 对于线性推理,不再封装,直接执行逻辑更清晰
batchBoxesType Engine::inferEngine(const pybind11::array &image) {
//    全部以多图情况处理
    pybind11::gil_scoped_release release;
    std::vector<cv::Mat> mats;
    cv::Mat mat(image.shape(1), image.shape(2), CV_8UC3, (unsigned char *) image.data(0));
    mats.emplace_back(mat);

    return inferEngine(mats);
}

batchBoxesType Engine::inferEngine(const std::vector<pybind11::array> &images) {
    pybind11::gil_scoped_release release;
    std::vector<cv::Mat> mats;
    std::cout<<"1"<<std::endl;
    for (auto &image: images) {
//      1. 预处理
        cv::Mat mat(image.shape(1), image.shape(2), CV_8UC3, (unsigned char *) image.data(0));
        mats.emplace_back(mat);
        std::cout<<"11"<<std::endl;
    }
    return inferEngine(mats);
}

batchBoxesType Engine::inferEngine(const cv::Mat &mat) {
    std::vector<cv::Mat> mats = {mat};

    return inferEngine(mats);
}

batchBoxesType Engine::inferEngine(const std::vector<cv::Mat> &images) {
//    pybind11::gil_scoped_release release;
    batchBoxes.clear();
    int inferNum = param->yoloDetectParam.batchSize;
    int countPre = 0;
    auto lastElement = &images.back();

    for (auto &image: images) {
//      1. 预处理
//        cv::Mat mat(image.shape(1), image.shape(2), CV_8UC3, (unsigned char *) image.data(0));
        std::cout<<"2"<<std::endl;
        curAlg->preProcess(param->yoloDetectParam, image, pinMemoryIn + countPre * singleInputSize);
        param->yoloDetectParam.d2is.push_back(
                {param->yoloDetectParam.d2i[0], param->yoloDetectParam.d2i[1], param->yoloDetectParam.d2i[2],
                 param->yoloDetectParam.d2i[3], param->yoloDetectParam.d2i[4], param->yoloDetectParam.d2i[5]}
        );
        for (int i = 0; i < 6; ++i) {
            std::cout<<param->yoloDetectParam.d2i[i]<<" ";
        }
        countPre += 1;
//      2. 判断推理数量
        // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
        if (&image != lastElement && countPre < param->yoloDetectParam.batchSize) continue;
        // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
        if (&image == lastElement) inferNum = countPre;

//      3. 执行推理
//       内存->gpu, 推理, 结果gpu->内存
        checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInSize, cudaMemcpyHostToDevice, commitStream));
//        对于pybind11::array类型图片,若在python中已做好预处理,不需要C++中处理什么, 可通过以下操作直接复制到gpu上,不需要pinMemoryIn中间步骤
//        checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInSize, cudaMemcpyHostToDevice, commitStream));
        param->yoloDetectParam.context->enqueueV3(commitStream);
        cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutSize, cudaMemcpyDeviceToHost, commitStream);

        countPre = 0;
        checkRuntime(cudaStreamSynchronize(commitStream));

//      4. 后处理
//        curFunc->postProcess(curParam, pinMemoryOut, singleOutputSize, outPost.inferNum, batchBox);
        curAlg->postProcess(param->yoloDetectParam, pinMemoryOut, singleOutputSize, inferNum, batchBox);
//       将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());
        param->yoloDetectParam.d2is.clear();
        batchBox.clear();
    }

    return batchBoxes;
}

int Engine::releaseEngine() {
    delete param;
//    delete data;
    logSuccess("Release engine success");
}

Engine::~Engine() {
    checkRuntime(cudaFree(gpuIn));
    checkRuntime(cudaFree(gpuOut));

    checkRuntime(cudaFreeHost(pinMemoryIn));
    checkRuntime(cudaFreeHost(pinMemoryOut));

//    checkRuntime(cudaStreamDestroy(commitStream));
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
//    pybind11::class_<batchBoxesType>(m, "batchBoxesType");
//            .def("get", &batchBoxesType::get);

//    暴露的推理接口
    pybind11::class_<Engine>(m, "Engine")
            .def(pybind11::init<>())
            .def("initEngine", &Engine::initEngine)

//            .def("inferEngine", pybind11::overload_cast<const std::string &>(&Engine::inferEngine))
//            .def("inferEngine", pybind11::overload_cast<const std::vector<std::string> &>(&Engine::inferEngine))
            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine), pybind11::arg("image"))
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine),
                 pybind11::arg("images"))

            .def("releaseEngine", &Engine::releaseEngine);
}