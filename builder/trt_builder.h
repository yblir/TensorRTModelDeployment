//
// Created by Administrator on 2023/7/21.
//

#ifndef TENSORRTMODELDEPLOYMENT_TRT_BUILDER_H
#define TENSORRTMODELDEPLOYMENT_TRT_BUILDER_H

#include <iostream>
#include "../base_infer/selfDataType.hpp"

namespace TRT {
//    enum class Mode : int {
//        FP32,
//        FP16
//    };
//    构建引擎文件,并保存到硬盘, 所有模型构建引擎文件方法都一样,如果加自定义层,继承算法各自实现
    bool compile(
            Mode mode,
            unsigned int maxBatchSize,
            const std::string &onnxFilePath,
            const std::string &saveEnginePath,
//            工作空间大小影响编译engine的速度, 代码中缺省, 在此设置. 若模型大小大于工作空间, 则编译会失败.
            size_t maxWorkspaceSize = 1ul << 29  // 512M
    );

    std::vector<unsigned char> loadEngine(const std::string &enginePath);
    std::vector<unsigned char> getEngine(BaseParam &param);
};

#endif //TENSORRTMODELDEPLOYMENT_TRT_BUILDER_H
