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

    curAlgParam->mode = inputParam.fp32 ? Mode::FP32 : Mode::FP16;

    curAlgParam->onnxPath = inputParam.onnxPath;
    curAlgParam->enginePath = inputParam.enginePath;

    curAlgParam->inputName = inputParam.inputName;
    curAlgParam->outputName = inputParam.outputName;

//   todo 参数挤在一起没区分度, 单独对trt制作一个参数?
    curAlgParam->trt = createInfer(*curAlg, *curAlgParam);
    if (nullptr == curAlgParam->trt) {
        return -1;
    }

//	  初始化时,创建batchSize个预处理,后处理函数线程, 一步完成预处理操作
    preExecutor.addThread(static_cast<unsigned short>(curAlgParam->batchSize));
    postExecutor.addThread(static_cast<unsigned short>(curAlgParam->batchSize));

//    // 更新trt context内容, 并计算trt推理需要的输入输出内存大小
//  memory 0:输入,1:输出, 记录的是batch个输入输出元素个数, 要换算成占用空间,需要乘上sizeof(float)
    auto memory = curAlgParam->trt->getMemory();
//    占用实际内存
    trtInMemorySize = memory[0] * sizeof(float);
    trtOutMemorySize = memory[1] * sizeof(float);
//    内存中单个输入输出元素数量, 用于计算向下一个内存单元递增时的索引位置
    singleInputSize = memory[0] / curAlgParam->batchSize;
    singleOutputSize = memory[1] / curAlgParam->batchSize;

    //  todo 对仿射参数的初始化,用于占位, 后续使用多线程预处理, 图片参数根据传入次序在指定位置插入
    for (int i = 0; i < curAlgParam->batchSize; ++i) {
        curAlgParam->preD2is.push_back({0., 0., 0., 0., 0., 0.});
    }

    checkRuntime(cudaMallocAsync(&gpuIn, trtInMemorySize, commitStream));
    checkRuntime(cudaMallocAsync(&gpuOut, trtOutMemorySize, commitStream));
    checkRuntime(cudaMallocHost(&pinMemoryIn, trtInMemorySize));
    checkRuntime(cudaMallocHost(&pinMemoryOut, trtOutMemorySize));

    curAlgParam->context->setTensorAddress(curAlgParam->inputName.c_str(), gpuIn);
    curAlgParam->context->setTensorAddress(curAlgParam->outputName.c_str(), gpuOut);

    cudaStreamSynchronize(commitStream);

    return 0;
}
/* todo 弃用
// 把pybind11::array转为cv::Mat再推理, 似乎效率有点低, 因为多了一步类型转换
batchBoxesType Engine::inferEngine(const pybind11::array &image) {
//    全部以多图情况处理
    pybind11::gil_scoped_release release;
    std::vector<cv::Mat> mats;
    cv::Mat mat(image.shape(0), image.shape(1), CV_8UC3, (unsigned char *) image.data(0));
    mats.emplace_back(mat);

    return inferEngine(mats);
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
*/

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
//   在流同步过程中赋值,此处引入postD2is,是为了保持和多线程处理一致, 因为具体处理函数,如YoloDetect.cpp中用的就是postD2is
    curAlgParam->postD2is = curAlgParam->preD2is;
//    5.流同步
    checkRuntime(cudaStreamSynchronize(commitStream));
//    6.后处理, 如果有需要的的话
    curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, 1, batchBox);
    return batchBox;
}

batchBoxesType Engine::inferEngine(const std::vector<pybind11::array> &images) {
//    pybind11::gil_scoped_release release;
    batchBoxes.clear();
//    inferNum 这里仅设置一个初始值, 如果实际推理数量!=batchSize,代码中会自动调整
    int inferNum = curAlgParam->batchSize;
    int preIndex = 0;
    auto lastElement = &images.back();

    for (auto &image: images) {
//      1. 预处理
//        curAlg->preProcess(curAlgParam, image, pinMemoryIn + preIndex * singleInputSize);
        auto curPinMemoryInIndex = pinMemoryIn + preIndex * singleInputSize;
//		线程池操作
        preThreadFlags.emplace_back(
                preExecutor.commit(
                        [this, image, curPinMemoryInIndex, preIndex] {
                            curAlg->preProcess(curAlgParam, image, curPinMemoryInIndex, preIndex);
                        })
        );
        preIndex += 1;
//      2. 判断推理数量
        // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
        if (&image != lastElement && preIndex < curAlgParam->batchSize) continue;
        // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
        if (&image == lastElement) inferNum = preIndex;

//		线程池处理方式, 所有预处理线程都完成后再继续
        for (auto &flag: preThreadFlags) {
            flag.get();
        }

//      3. 执行推理
//        对于pybind11::array类型图片,若在python中已做好预处理,不需要C++中处理什么, 可通过以下操作直接复制到gpu上,不需要pinMemoryIn中间步骤
//        checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//       内存->gpu, 推理, 结果gpu->内存
        checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));

        curAlgParam->context->enqueueV3(commitStream);
        cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream);

//      thread_flags是存储线程池状态的vector, 每次处理完后都清理, 不然会越堆越多,内存泄露
        preThreadFlags.clear();
//        在流同步过程中赋值,此处引入postD2is,是为了保持和多线程处理一致, 因为具体处理函数,如YoloDetect.cpp中用的就是postD2is
        curAlgParam->postD2is = curAlgParam->preD2is;
        preIndex = 0;

        checkRuntime(cudaStreamSynchronize(commitStream));

//      4. 后处理
//        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, inferNum, batchBox);
//        多线程后处理似乎并不快, 因为先把处理结果存储在字典boxDict中,再把结果push到vector batchBox中, 转换过程同样耗时. 如果没有
//        后处理,推理结果仅仅是二分类之类的概率值, 多线程处理反而更慢, 此时直接用上面的循环postProcess取出结果更快.
// ---------------------------------------------------------------------------------------------------------------------
        for (int i = 0; i < inferNum; ++i) {
            postThreadFlags.emplace_back(
                    postExecutor.commit([this, i] {
//                        为batch中每个元素进行单个处理后处理,一步获得所有处理结果
                        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, boxDict, i);
                    })
            );
        }

        //	线程池处理方式, 所有预处理线程都完成后再继续
        for (auto &flag: postThreadFlags) {
            flag.get();
        }
        postThreadFlags.clear();

        for (int i = 0; i < inferNum; ++i) {
            batchBox.push_back(boxDict[i]);
        }
// ---------------------------------------------------------------------------------------------------------------------
//       将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());

        batchBox.clear();
    }

    return batchBoxes;
}


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
    int preIndex = 0;
    auto lastElement = &images.back();

    for (auto &image: images) {
//      1. 预处理
//        curAlg->preProcess(curAlgParam, image, pinMemoryIn + preIndex * singleInputSize);
        auto curPinMemoryInIndex = pinMemoryIn + preIndex * singleInputSize;

//		taskflow的多线程方式, 会有内存越界问题, 不好用		
//        tasks.emplace_back(taskflow.emplace([this, image, aa]() {
//            curAlg->preProcess(curAlgParam, image, curPinMemoryInIndex);
//        }));

//		  原生多线程处理方式, 每次预处理都要创建线程,然后销毁, 效率应该不高, 但实测与线程池没太大差别
//        threads.emplace_back([this, image, aa, preIndex]() {
//            curAlg->preProcess(curAlgParam, image, curPinMemoryInIndex, preIndex);
//        });

//		线程池操作
        preThreadFlags.emplace_back(
                preExecutor.commit([this, image, curPinMemoryInIndex, preIndex] {
                    curAlg->preProcess(curAlgParam, image, curPinMemoryInIndex, preIndex);
                })
        );
        preIndex += 1;
//      2. 判断推理数量
        // 不是最后一个元素且数量小于batchSize,继续循环,向pinMemoryIn写入预处理后数据
        if (&image != lastElement && preIndex < curAlgParam->batchSize) continue;
        // 若是最后一个元素,记录当前需要推理图片数量(可能小于一个batchSize)
        if (&image == lastElement) inferNum = preIndex;
/*
        // 等待所有线程完成,原生线程处理方式
        for (auto &thread: threads) {
            if (thread.joinable()) thread.join();
        }
        threads.clear();
*/
//		线程池处理方式, 所有预处理线程都完成后再继续
        for (auto &flag: preThreadFlags) {
            flag.get();
        }

//      3. 执行推理
//        对于pybind11::array类型图片,若在python中已做好预处理,不需要C++中处理什么, 可通过以下操作直接复制到gpu上,不需要pinMemoryIn中间步骤
//        checkRuntime(cudaMemcpyAsync(gpuIn, image.data(0), trtInMemorySize, cudaMemcpyHostToDevice, commitStream));
//       内存->gpu, 推理, 结果gpu->内存
        checkRuntime(cudaMemcpyAsync(gpuIn, pinMemoryIn, trtInMemorySize, cudaMemcpyHostToDevice, commitStream));

        curAlgParam->context->enqueueV3(commitStream);
        cudaMemcpyAsync(pinMemoryOut, gpuOut, trtOutMemorySize, cudaMemcpyDeviceToHost, commitStream);
//      thread_flags是存储线程池状态的vector, 每次处理完后都清理, 不然会越堆越多,内存泄露
        preThreadFlags.clear();
//        在流同步过程中赋值,此处引入postD2is,是为了保持和多线程处理一致, 因为具体处理函数,如YoloDetect.cpp中用的就是postD2is
        curAlgParam->postD2is = curAlgParam->preD2is;
        preIndex = 0;
        checkRuntime(cudaStreamSynchronize(commitStream));

//      4. 后处理
        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, inferNum, batchBox);
//      todo 多线程后处理似乎并不快, 因为先把处理结果存储在字典boxDict中,再把结果push到vector batchBox中, 转换过程同样耗时. 如果没有
//      todo 后处理,推理结果仅仅是二分类之类的概率值, 多线程处理反而更慢, 此时直接用上面的循环postProcess取出结果即可.
// ---------------------------------------------------------------------------------------------------------------------
//        for (int i = 0; i < inferNum; ++i) {
//            postThreadFlags.emplace_back(
//                    postExecutor.commit([this, i] {
////                        为batch中每个元素进行单个处理后处理,一步获得所有处理结果
//                        curAlg->postProcess(curAlgParam, pinMemoryOut, singleOutputSize, boxDict, i);
//                    })
//            );
//        }
//
//        //	线程池处理方式, 所有预处理线程都完成后再继续
//        for (auto &flag: postThreadFlags) {
//            flag.get();
//        }
//        postThreadFlags.clear();
//
//        for (int i = 0; i < inferNum; ++i) {
//            batchBox.push_back(boxDict[i]);
//        }
// ---------------------------------------------------------------------------------------------------------------------
//       将每次后处理结果合并到输出vector中
        batchBoxes.insert(batchBoxes.end(), batchBox.begin(), batchBox.end());

        batchBox.clear();
    }

    return batchBoxes;
}

int Engine::releaseEngine() {
    delete curAlg;
    delete curAlgParam;

    logSuccess("Release engine success");
}

Engine::~Engine() {
    checkRuntime(cudaFree(gpuIn));
    checkRuntime(cudaFree(gpuOut));

    checkRuntime(cudaFreeHost(pinMemoryIn));
    checkRuntime(cudaFreeHost(pinMemoryOut));
}

PYBIND11_MODULE(deployment, m) {
//    配置手动输入参数
    pybind11::class_<ManualParam>(m, "ManualParam")
            .def(pybind11::init<>())
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
//    pybind11::class_<batchBoxesType>(m, "batchBoxesType");
//            .def("get", &batchBoxesType::get);

//    暴露的推理接口
    pybind11::class_<Engine>(m, "Engine")
            .def(pybind11::init<>())
            .def("initEngine", &Engine::initEngine)

//          todo 在多线程操作时, 必须加上pybind11::call_guard<pybind11::gil_scoped_release>()才能运行,单线程不加也可以,目前没发现加与不加的区别,难道
//          todo 会影响运行速度?, 没测过!
            .def("inferEngine", pybind11::overload_cast<const pybind11::array &>(&Engine::inferEngine),
                 pybind11::arg("image"), pybind11::call_guard<pybind11::gil_scoped_release>())
            .def("inferEngine", pybind11::overload_cast<const std::vector<pybind11::array> &>(&Engine::inferEngine),
                 pybind11::arg("images"), pybind11::call_guard<pybind11::gil_scoped_release>())

            .def("releaseEngine", &Engine::releaseEngine);
}