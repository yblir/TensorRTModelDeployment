//
// Created by Administrator on 2023/1/9.
//
#include <cuda_runtime.h>
#include "single_interface.h"


int Engine::initEngine(ManualParam &inputParam) {
//    param = new productParam;

    curAlg = new YoloDetect();
    curAlgParam = new YoloDetectParam;

    if (inputParam.batchSize > curAlgParam->maxBatch) {
        logError("input batchSize more than maxBatch of built engine, the maxBatch is 16");
        return -1;
    }

//    inputParam结构体参数从python中传入
    curAlgParam->gpuId = inputParam.gpuId;
    curAlgParam->batchSize = inputParam.batchSize;

    curAlgParam->inputHeight = inputParam.inputHeight;
    curAlgParam->inputWidth = inputParam.inputWidth;

    curAlgParam->mode = inputParam.fp16 ? Mode::FP16 : Mode::FP32;

    curAlgParam->onnxPath = inputParam.onnxPath;
    curAlgParam->enginePath = inputParam.enginePath;

    curAlgParam->inputName = inputParam.inputName;
    curAlgParam->outputName = inputParam.outputName;

//   todo 参数挤在一起没区分度, 单独对trt制作一个参数?
    curAlgParam->trt = createInfer(*curAlg, *curAlgParam);
    if (nullptr == curAlgParam->trt) {
        return -1;
    }

//    // 更新trt context内容, 并计算trt推理需要的输入输出内存大小
//  memory 0:输入,1:输出, 记录的是batch个输入输出元素个数, 要换算成占用空间,需要乘上sizeof(float)
    auto memory = curAlgParam->trt->getMemory();
//    占用实际内存
    trtInMemorySize = memory[0] * sizeof(float);
    trtOutMemorySize = memory[1] * sizeof(float);
//    内存中单个输入输出元素数量, 用于计算向下一个内存单元递增时的索引位置
    singleInputSize = memory[0] / curAlgParam->batchSize;
    singleOutputSize = memory[1] / curAlgParam->batchSize;

    checkRuntime(cudaMallocAsync(&gpuIn, trtInMemorySize, commitStream));
    checkRuntime(cudaMallocAsync(&gpuOut, trtOutMemorySize, commitStream));
    checkRuntime(cudaMallocHost(&pinMemoryIn, trtInMemorySize));
    checkRuntime(cudaMallocHost(&pinMemoryOut, trtOutMemorySize));

    curAlgParam->context->setTensorAddress(curAlgParam->inputName.c_str(), gpuIn);
    curAlgParam->context->setTensorAddress(curAlgParam->outputName.c_str(), gpuOut);

    cudaStreamSynchronize(commitStream);

    return 0;
}

//// 把pybind11::array转为cv::Mat再推理, 似乎效率有点低, 因为多了一步类型转换
//batchBoxesType Engine::inferEngine(const pybind11::array &image) {
////    全部以多图情况处理
//    pybind11::gil_scoped_release release;
//    std::vector<cv::Mat> mats;
//    cv::Mat mat(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));
//    mats.emplace_back(mat);
//
//    return inferEngine(mats);
//}

// 对于单张单线程推理, 直接拷贝到GPU上推理, 不需要中间过程
batchBoxesType Engine::inferEngine(const pybind11::array &image) {
//    pybind11::gil_scoped_release release;
//    batchBox是类成员变量, 不会自动释放, 手动情况上一次推理的结果
    batchBox.clear();
//    1.预处理
    curAlg->preProcess(curAlgParam, image, pinMemoryIn);

//    2.待推理数据异步拷贝到gpu上
//    如果不在c++中做预处理, 可以直接拷贝的到gpu上
//    checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//   如果在c++中做了预处理, 预处理后的数据在锁叶内存pinMemoryIn中, 需要从pinMemoryIn拷贝到gpu上
    checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));

//    3.异步推理
    curAlgParam->context->enqueueV3(commitStream);
//    4.推理结果gpu异步拷贝内存中
    checkRuntime(cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream));
//    5.流同步
    checkRuntime(cudaStreamSynchronize(commitStream));
//    6.后处理, 如果有需要的的话
    curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, 1, batchBox);
    return batchBox;
}

batchBoxesType Engine::inferEngine(const std::vector<pybind11::array> &images) {
//    pybind11::gil_scoped_release release;
    batchBoxes.clear();
    int inferNum = curAlgParam->batchSize;
    int countPre = 0;
    auto lastElement = &images.back();

    for (auto &image: images) {
//      1. 预处理
        curAlg->preProcess(curAlgParam, image, pinMemoryIn + countPre * singleInputSize);

        countPre += 1;
//      2. 判断推理数量
        // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
        if (&image != lastElement && countPre < curAlgParam->batchSize) continue;
        // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
        if (&image == lastElement) inferNum = countPre;

//      3. 执行推理
//        对于pybind11::array类型图片,若在python中已做好预处理,不需要C++中处理什么, 可通过以下操作直接复制到gpu上,不需要pinMemoryIn中间步骤
//        checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//       内存->gpu, 推理, 结果gpu->内存
        checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));

        curAlgParam->context->enqueueV3(commitStream);
        cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream);

        countPre = 0;
        checkRuntime(cudaStreamSynchronize(commitStream));

//      4. 后处理
        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, inferNum, batchBox);
//       将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());

        batchBox.clear();
    }

    return batchBoxes;
}

//batchBoxesType Engine::inferEngine(const std::vector<pybind11::array> &images) {
//    pybind11::gil_scoped_release release;
//    std::vector<cv::Mat> mats;
//    for (auto &image: images) {
////      1. 预处理
//        cv::Mat mat(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));
//        mats.emplace_back(mat);
//    }
//    return inferEngine(mats);
//}

//batchBoxesType Engine::inferEngine(const cv::Mat &mat) {
//    std::vector<cv::Mat> mats = {mat};
//
//    return inferEngine(mats);
//}

batchBoxesType Engine::inferEngine(const cv::Mat &mat) {
    batchBox.clear();
    curAlg->preProcess(curAlgParam, mat, pinMemoryIn);

//    1.待推理数据异步拷贝到gpu上
//   如果在c++中做了预处理, 预处理后的数据在锁页内存pinMemoryIn中, 需要从pinMemoryIn拷贝到gpu上
    checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//    2.异步推理
    curAlgParam->context->enqueueV3(commitStream);
//    3.推理结果gpu异步拷贝内存中
    checkRuntime(cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream));
//    4.流同步
    checkRuntime(cudaStreamSynchronize(commitStream));
//    5.后处理, 如果有需要的的话
    curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, 1, batchBox);

    return batchBox;
}

batchBoxesType Engine::inferEngine(const std::vector<cv::Mat> &images) {
//    pybind11::gil_scoped_release release;
    batchBoxes.clear();
    int inferNum = curAlgParam->batchSize;
    int countPre = 0;
    auto lastElement = &images.back();

    for (auto &image: images) {
//      1. 预处理
        curAlg->preProcess(curAlgParam, image, pinMemoryIn + countPre * singleInputSize);

        countPre += 1;
//      2. 判断推理数量
        // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
        if (&image != lastElement && countPre < curAlgParam->batchSize) continue;
        // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
        if (&image == lastElement) inferNum = countPre;

//      3. 执行推理
//        对于pybind11::array类型图片,若在python中已做好预处理,不需要C++中处理什么, 可通过以下操作直接复制到gpu上,不需要pinMemoryIn中间步骤
//        checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//       内存->gpu, 推理, 结果gpu->内存
        checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));

        curAlgParam->context->enqueueV3(commitStream);
        cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream);

        countPre = 0;
        checkRuntime(cudaStreamSynchronize(commitStream));

//      4. 后处理
        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, inferNum, batchBox);
//       将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());

        batchBox.clear();
    }

    return batchBoxes;
}

int Engine::releaseEngine() {
    delete curAlg;
    delete curAlgParam;
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

//          todo 在多线程操作时, 必须加上pybind11::call_guard<pybind11::gil_scoped_release>()才能运行,单线程不加也可以,目前没发现加与不加的区别,难道
//          todo 会影响运行速度?, 没测过!
            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine),
                 pybind11::arg("image"), pybind11::call_guard<pybind11::gil_scoped_release>())
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine),
                 pybind11::arg("images"), pybind11::call_guard<pybind11::gil_scoped_release>())

            .def("releaseEngine", &Engine::releaseEngine);
}