//
// Created by Administrator on 2023/1/9.
//

#include "face_interface.h"

int initEngine(Handle &engine, struct  productConfig &conf) {
    //人脸检测模型初始化

    if (conf.yoloFace == nullptr) {
        bool is_ok = AlgorithmFactory::loadAlgorithmSo("fdfsdfdfds");
        if (!is_ok) printf("error");

        conf.yoloFace->initParam(conf.yoloFace);
        bool flag=conf.yoloFace->loadEngine("dffsd");
        if (!flag){
            bool flag2=conf.yoloFace->buildEngine("sd","df");
            if (flag2)
                bool flag3=conf.yoloFace->loadEngine("dffsd");
            else
                printf("error");
        }
    }

    // 其他检测模型初始化
    if(conf.yoloFace== nullptr){
        "fsdfdsfsdf";
    }
}


