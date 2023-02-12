//
// Created by Administrator on 2023/1/9.
//

#include "face_interface.h"

int initEngine(Handle &engine, struct productConfig &conf) {
    //人脸检测模型初始化
    if (conf.yoloFace == nullptr) {
        bool is_ok = AlgorithmFactory::loadDynamicLibrary("fdfsdfdfds");
        if (!is_ok) printf("error");

        conf.yoloFace->initParam(conf.yoloFace);

        // 判断引擎文件是否存在,如果不存在,要先构建engine
        if (std::filesystem::exists(conf.yoloFace->conf2.enginePath))
            std::vector<unsigned char> engineFile = YoloFace::loadEngine("dffsd");
        else {
            bool flag2 = YoloFace::buildEngine(conf.yoloFace->conf2.onnxPath, conf.yoloFace->conf2.enginePath, 1);
            if (flag2) std::vector<unsigned char> engineFile = YoloFace::loadEngine("dffsd");
        }
    }

    // 其他检测模型初始化
    if (conf.yoloFace == nullptr) {
        "fsdfdsfsdf";
    }
}


