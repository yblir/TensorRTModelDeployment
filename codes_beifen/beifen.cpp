//
// Created by Administrator on 2023/2/22.
//

/*
//使用引擎推理图片
//int inferEngine(Handle engine, unsigned char *imgData, int imgWidth, int imgHeight, int min_face_size, int mode,
//                int imgPixelFormat, int &res_num) {
//int inferEngine(struct productConfig &conf, struct productFunc &func, cv::Mat &image, struct productOutput out) {
int inferEngine_back(struct productConfig &conf, struct productFunc &func, std::vector<cv::Mat> &matVector,
                     struct productOutput out) {

//    func.yoloFace->preProcess(image);

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();
    //
    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
//    //计算输入tensor所占存储空间大小
//    int inputSize = conf.detectConfig.batchSize * inputChannel * inputHeight * inputWidth;
//
//    // 在锁页内存和gpu上开辟输入tensor数据所在存储空间
//    checkRuntime(cudaMallocHost(&pinMemoryIn, inputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryIn, inputSize * sizeof(float)));

    // 填充灰边, 缩放图片到模型输入指定的尺寸
    cv::Mat scaleImage = letterBox(image, inputWidth, inputHeight, d2i1);

//    BGR2RGB(scaleImage, pinMemoryIn);

    //模型输入数据从锁页内存转到gpu上
    checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));

    //获得输入tensor形状,设置指定的动态batch的大小,之后再重新指定输入tensor形状
    auto inputShape = conf.detectConfig.engine->getTensorShape(conf.detectConfig.inputName.c_str());
    inputShape.d[0] = conf.detectConfig.batchSize;
    context->setInputShape(conf.detectConfig.inputName.c_str(), inputShape);

//    // 获得输出tensor形状,计算输出所占存储空间
//    auto outputShape = conf.detectConfig.engine->getTensorShape(conf.detectConfig.outputName.c_str());
////    int boxNum = outputShape.d[1];
////    int predictNum = outputShape.d[2];
////    int classNum = predictNum - 5;
//    int outputSize = conf.detectConfig.batchSize * outputShape.d[1] * outputShape.d[2];
//
//    // 分别在锁页内存和gpu上开辟空间,用于存储推理结果
//    checkRuntime(cudaMallocHost(&pinMemoryOut, outputSize * sizeof(float)));
//    checkRuntime(cudaMalloc(&gpuMemoryOut, outputSize * sizeof(float)));


    // 指定onnx中输入输出tensor名
    context->setTensorAddress(conf.detectConfig.inputName.c_str(), gpuMemoryIn);
    context->setTensorAddress(conf.detectConfig.outputName.c_str(), gpuMemoryOut);

    // 执行异步推理
    context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinMemoryOut, gpuMemoryOut, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);

    return 0;
}

int inferEngine_back2(struct productConfig &conf, struct productFunc &func, std::vector<cv::Mat> &matVector,
                      struct productOutput out) {

    //创建cuda任务流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();

    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
    //0:输入存储空间大小,1:输出存储空间大小
    std::vector<int> memory = setBatchAndInferMemory(conf.detectConfig, context, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);

//    std::vector<float *> d2i;
    // 预处理,一次处理batch张图片, 包括尺寸缩放,归一化,色彩转换,图片数据从内存提取到gpu
    func.yoloDetect->preProcess(image);
    // 填充灰边, 缩放图片到模型输入指定的尺寸
//    cv::Mat scaleImage = letterBox(image, conf.detectConfig.inputWidth, conf.detectConfig.inputHeight, d2i1);
    cv::Mat scaleImage = letterBox(image, conf.detectConfig.inputWidth, conf.detectConfig.inputHeight);
//    BGR2RGB(scaleImage, pinMemoryIn);

    //模型输入数据从锁页内存转到gpu上
    checkRuntime(cudaMemcpyAsync(gpuMemoryIn, pinMemoryIn, memory[0] * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 指定onnx中输入输出tensor名
    context->setTensorAddress(conf.detectConfig.inputName.c_str(), gpuMemoryIn);
    context->setTensorAddress(conf.detectConfig.outputName.c_str(), gpuMemoryOut);
    // 执行异步推理
    context->enqueueV3(stream);

    // 将推理结果从gpu拷贝到cpu上
    cudaMemcpyAsync(pinMemoryOut, gpuMemoryOut, memory[1] * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 流同步
    cudaStreamSynchronize(stream);

    //后处理
    func.yoloFace->postProcess(image);

    return 0;
}

int inferEngine_back3(struct productConfig &conf, struct productFunc &func,
                      std::vector<cv::Mat> &matVector, struct productOutput out) {
    // 创建上下文管理器
    nvinfer1::IExecutionContext *context = conf.detectConfig.engine->createExecutionContext();

    //配置锁页内存,gpu显存指针
    float *pinMemoryIn = nullptr, *pinMemoryOut = nullptr, *gpuMemoryIn = nullptr, *gpuMemoryOut = nullptr;
    //0:当前推理模型输入tensor存储空间大小,1:当前推理输出结果存储空间大小
    std::vector<int> memory = setBatchAndInferMemory(conf.detectConfig, context, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);

    // 预处理,一次处理batch张图片, 包括尺寸缩放,归一化,色彩转换,图片数据从内存提取到gpu
    int batch = 0;
    for (auto &mat: matVector) {
        // 遍历所有图片,若图片数量不够一个batch,加入的处理队列中
        if (batch < conf.detectConfig.batchSize) {
            func.yoloDetect->preProcess(mat, pinMemoryIn);
            batch += 1;
        } else {    //够一个batchSize,执行推理
            trtInferProcess(conf.detectConfig, context, memory, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);
            //后处理
            func.yoloFace->postProcess(image);
            // 清空标记,重新开始下一个batch.
            batch = 0;
        }
    }

    // batch不为0,说明最后的图片不够一个batchSize,没有推理就退出了.这里要把剩余的图片继续推理了!
    if (0 != batch) {
        trtInferProcess(conf.detectConfig, context, memory, pinMemoryIn, pinMemoryOut, gpuMemoryIn, gpuMemoryOut);
        //后处理
        func.yoloFace->postProcess(image);
    }

    return 0;
}
*/