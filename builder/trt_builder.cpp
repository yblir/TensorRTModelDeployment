//
// Created by Administrator on 2023/7/21.
//
//#include <fstream>
#include <filesystem>
#include <NvOnnxParser.h>
#include <fstream>

#include "../utils/general.h"
#include "trt_builder.h"

bool TRT::compile(
        Mode mode,
        unsigned int maxBatchSize,
        const std::string &onnxFilePath,
        const std::string &saveEnginePath,
        size_t maxWorkspaceSize // 256M
) {
    //检查待转换的onnx文件是否存在
    if (!std::filesystem::exists(onnxFilePath)) {
        std::cout << "onnx path not exist: " << onnxFilePath << std::endl;
        return false;
    }

    TRTLogger logger;
    uint32_t flag = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //1 基本组件, 不想麻烦就就全部写成auto类型~
    auto builder = ptrFree(nvinfer1::createInferBuilder(logger));
    auto config = ptrFree(builder->createBuilderConfig());
    std::shared_ptr<nvinfer1::INetworkDefinition> network = ptrFree(builder->createNetworkV2(flag));

//    在config中配置化fp16
    if (mode == Mode::FP16) {
        if (!builder->platformHasFastFp16()) {
//            INFOW("Platform not have fast fp16 support");
            printf("Platform not have fast fp16 support");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    //2 通过onnxparser解析器的结果,填充到network中,类似addconv的方式添加进去
    auto parser = ptrFree(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnxFilePath.c_str(), 1)) {
        printf("failed to parse onnx file\n");
        return false;
    }

    //3 设置最大工作缓存
    printf("Workspace Size = %.2f MB\n", float(maxWorkspaceSize) / 1024.0f / 1024.0f);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, maxWorkspaceSize);


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
    std::shared_ptr<nvinfer1::IHostMemory> serializedModel = ptrFree(builder->buildSerializedNetwork(*network, *config));
    if (nullptr == serializedModel) {
        printf("build engine failed\n");
        return false;
    }

    //6 将推理引擎序列化,存储为文件
    auto *f = fopen(saveEnginePath.c_str(), "wb");
    fwrite(serializedModel->data(), 1, serializedModel->size(), f);
    fclose(f);

    printf("build and save engine success \n");

    return true;
}

//从硬盘加载引擎文件到内存
std::vector<unsigned char> TRT::loadEngine(const std::string &enginePath) {
    //检查engine文件是否存在
    if (!std::filesystem::exists(enginePath)) {
        std::cout << "engine path not exist: " << enginePath << std::endl;
        return {};
    }

    std::ifstream in(enginePath, std::ios::in | std::ios::binary);
    if (!in.is_open()) { return {}; }

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *) &data[0], length);
    }
    in.close();

    return data;
}


std::vector<unsigned char> TRT::getEngine(const std::string &enginePath, const BaseParam &param) {
    std::vector<unsigned char> engine;

    // 判断引擎文件是否存在,如果不存在,要先构建engine
    if (std::filesystem::exists(enginePath))
        // engine存在,直接加载engine文件,反序列化引擎文件到内存
        engine = loadEngine(enginePath);
    else {
        //engine不存在,先build,序列化engine到硬盘, 再执行反序列化操作
        if (compile(param.mode, param.batchSize, param.onnxPath, param.enginePath, 1 << 28)) // 256M
            engine = loadEngine(enginePath);
    }
    return engine;
}
