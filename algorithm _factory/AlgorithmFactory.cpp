//
// Created by 12134 on 2023/2/9.
//

#include "AlgorithmFactory.h"

bool AlgorithmFactory::buildEngine(const std::string &onnxFilePath,const std::string &saveEnginePath) {
    if (fileExist(onnxFilePath)) {
        printf("yolov5s.trt has exist.\n");
        return true;
    }

    TRTLogger logger;

    //1 基本组件
    auto builder = prtFree(nvinfer1::createInferBuilder(logger));
    auto config = prtFree(builder->createBuilderConfig());
    auto network = prtFree(builder->createNetworkV2(1));

    //2 通过onnxparser解析器的结果,填充到network中,类似addconv的方式添加进去
    auto parser = prtFree(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(
            "/mnt/e/ClionCpp/tensorRT_learning2/tensorrt-integrate-1.2-yolov5-detect/workspace/yolov5s.onnx", 1)) {
        printf("failed to parse faceyolo.onnx\n");
        //自动释放了
        return false;
    }

    //3 设置最大工作缓存
    int maxBatchSize = 1;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    //4 若模型有多个输入,必须使用多个profile,好在一般情况想下都是1
    auto profile = builder->createOptimizationProfile();

    auto inputTensor = network->getInput(0);  //取得第一个输入,一般输入batch都是1,0号索引必定能取得,1号索引可以不?
    auto inputDims = inputTensor->getDimensions();

    //配置最小最大范围, 目标检测领域,输入尺寸是不变的,变化的batch
    inputDims.d[0] = 1;
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, inputDims);
    inputDims.d[0] = maxBatchSize;
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, inputDims);
    config->addOptimizationProfile(profile);

    //5 直接序列化推理引擎
    auto engine = prtFree(builder->buildSerializedNetwork(*network, *config));
    if (nullptr == engine) {
        printf("build engine build\n");
        return false;
    }

    //6 将推理引擎序列化,存储为文件
    auto *f = fopen("yolov5s.trt", "wb");
    fwrite(engine->data(), 1, engine->size(), f);
    fclose(f);

    printf("build done \n");
    return true;
}


bool AlgorithmFactory::loadEngine(const std::string &engineFilePath) {
    return false;
}

bool AlgorithmFactory::loadAlgorithmSo(const std::string &soPath) {
    return 0;
}
