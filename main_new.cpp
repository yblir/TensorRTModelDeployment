#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

#include <iostream>
#include <iostream>
#include <chrono>

#include <string>
#include<sys/time.h>
#include<unistd.h>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include <dirent.h>

#include "interface/face_interface_thread.h"
//#include "file_base.h"
#include "algorithm_factory/struct_data_type.h"
//#include "algorithm_product/product.h"
#include "utils/general.h"
#include "utils/box_utils.h"

// 0 /mnt/i/GitHub/TensorRTModelDeployment/imgs
int main(int argc, char *argv[]) {
    /*
    argc:参数个数
    *argv: 字符数组,记录输入的参数.可执行文件总在0号位,作为一个参数
    */
    // 判断参数个数, 若不为3,终止程序
    auto timer = new Timer();
    double total;
    if (argc != 3) {
        std::cout << " the number of param is incorrect, must be 3, but now is " << argc << std::endl;
        std::cout << "param format is ./AiSdkDemo gpu_id img_dir_path" << std::endl;
        return -1;
    }

    // =====================================================================
    // 外接传入的配置文件,和使用过程中生成的各种路径等
    struct productParam param;
    // 加{},说明创建的对象为nullptr, 存储从动态库解析出来的算法函数和类
    struct productFunc func{};
    struct productResult outs;
//    Handle engine;

//    conf.yoloConfig.onnxPath = "/mnt/e/GitHub/TensorRTModelDeployment/models/face_detect_v0.5_b17e5c7577192da3d3eb6b4bb850f8e_1out.onnx";
//    conf.yoloConfig.gpuId = int(strtol(argv[1], nullptr, 10));

//    param.yoloDetectParam.onnxPath = "/mnt/i/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
    param.yoloDetectParam.onnxPath = "/mnt/e/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
    param.yoloDetectParam.gpuId = int(strtol(argv[1], nullptr, 10));
    param.yoloDetectParam.batchSize = 2;
    param.yoloDetectParam.inputHeight = 640;
    param.yoloDetectParam.inputWidth = 640;
    param.yoloDetectParam.inputName = "images";
    param.yoloDetectParam.outputName = "output";
    param.yoloDetectParam.iouThresh = 0.5;
    param.yoloDetectParam.scoreThresh = 0.5;

    int ret = initEngine(param, func);
    if (ret != 0)
        return ret;
    std::cout << "init ok !" << std::endl;
    // ============================================================================================

    //创建输出文件夹
//    std::string path1 = std::string(argv[2]) + "/";
    std::string path1="/mnt/e/cartoon_data/personai_icartoonface_detval/";
//    std::string path1="/mnt/d/VOCdevkit/voc_test/";
    std::filesystem::path imgInputDir(path1);
    std::filesystem::path imgOutputDir(path1 + "output/");
    //检查文件夹路径是否合法, 检查输出文件夹路径是否存在,不存在则创建
    // 输入不是文件夹,或文件不存在抛出异常
    if (!std::filesystem::exists(imgInputDir) || !std::filesystem::is_directory(imgInputDir))
        return -1;
    //创建输出文件夹
    if (!std::filesystem::exists(imgOutputDir))
        std::filesystem::create_directories(imgOutputDir);

    std::vector<std::string> imagePaths;
    // 获取该文件夹下所有图片绝对路径,存储在vector向量中
    getImagePath(imgInputDir, imagePaths);
    auto t = timer->curTimePoint();
    inferEngine(param, func, imagePaths, outs);
    total = timer->timeCount(t);
    printf("total time: %.2f\n", total);
//    std::cout << "在原图上画框" << std::endl;
//    int i = 0;
//    // 画yolo目标检测框
//    if (!outs.detectResult.empty()) {
//        // 遍历每张图片
//        for (auto &out: outs.detectResult) {
//            if (out.empty()) {
//                i += 1;
//                continue;
//            }
//            cv::Mat img = cv::imread(imagePaths[i]);
//            // 遍历一张图片中每个预测框,并画到图片上
//            for (auto &box: out) {
//                drawImage(img, box);
//            }
//            // 把画好框的图片写入本地
//            std::string drawName = "draw" + std::to_string(i) + ".jpg";
//            cv::imwrite(imgOutputDir / drawName, img);
//            i += 1;
//        }
//    }
//    printf("right over!\n");
//    return 0;
}

//pre   use time: 23570.03 ms, thread use time: 275680.40 ms, pre img num = 10000
//infer use time: 48621.25 ms, thread use time: 275689.61 ms
//post  use time: 4344.31 ms, thread use time: 275692.25 ms
//total time: 275692.59 多线程
//total time: 107878.64 单线程

// 第二次
//total time: 140312.34 多线程

//pre   use time: 7296.06 ms, thread use time: 103052.07 ms, pre img num = 2000
//infer use time: 102836.99 ms, thread use time: 103712.01 ms
//==========================
//runtime error /mnt/i/GitHub/TensorRTModelDeployment/interface/face_interface_new.cpp:351  cudaFree(pinMemoryOut) failed.
//code = cudaErrorInvalidValue, message = invalid argument
//        post  use time: 1141.27 ms, thread use time: 103715.55 ms
//        total time: 103717.01

// total time: 150920.16

// total time: 72240.65 单线程
// total time: 36926.75 多线程

//pre   use time: 8733.38 ms, thread use time: 60971.82 ms, pre img num = 2000
//infer use time: 60714.26 ms, thread use time: 61155.07 ms
//runtime error /mnt/i/GitHub/TensorRTModelDeployment/interface/face_interface_new.cpp:343  cudaFreeAsync(pinMemoryOut, stream) failed.
//  code = cudaErrorInvalidValue, message = invalid argument
//post  use time: 2292.49 ms, thread use time: 61157.56 ms
//total time: 61159.27

// pre   use time: 7098.98 ms, thread use time: 41385.91 ms, pre img num = 2000
//infer use time: 28610.63 ms, thread use time: 41414.50 ms
//runtime error /mnt/i/GitHub/TensorRTModelDeployment/interface/face_interface_new.cpp:343  cudaFreeAsync(pinMemoryOut, stream) failed.
//  code = cudaErrorInvalidValue, message = invalid argument
//post  use time: 816.60 ms, thread use time: 41416.87 ms
//total time: 41418.07


//runtime error /mnt/e/GitHub/TensorRTModelDeployment/interface/face_interface_new.cpp:236  cudaFreeAsync(pinMemoryIn, stream) failed.
//  code = cudaErrorInvalidValue, message = invalid argument
//pre   use time: 23950.46 ms, thread use time: 433624.82 ms, pre img num = 10000
//infer use time: 59202.62 ms, thread use time: 433635.70 ms
//runtime error /mnt/e/GitHub/TensorRTModelDeployment/interface/face_interface_new.cpp:343  cudaFreeAsync(pinMemoryOut, stream) failed.
//  code = cudaErrorInvalidValue, message = invalid argument
//post  use time: 2646.69 ms, thread use time: 433637.90 ms
//total time: 433639.40
//total time: 218542.59


//pre   use time: 23609.07 ms, thread use time: 148098.20 ms, pre img num = 10000
//infer use time: 45088.74 ms, thread use time: 148107.23 ms
//runtime error /mnt/e/GitHub/TensorRTModelDeployment/interface/face_interface_thread.cpp:342  cudaFreeAsync(pinMemoryOut, stream) failed.
//  code = cudaErrorInvalidValue, message = invalid argument
//post  use time: 2915.31 ms, thread use time: 148109.52 ms
//total time: 148110.45

//pre   use time: 23572.81 ms, thread use time: 148410.35 ms, pre img num = 10000
//infer use time: 45030.09 ms, thread use time: 148419.60 ms
//        post  use time: 3054.38 ms, thread use time: 148424.88 ms
//        total time: 148426.40

//pre   use time: 23619.03 ms, thread use time: 147302.26 ms, pre img num = 10000
//infer use time: 44853.61 ms, thread use time: 147311.58 ms
//        post  use time: 3060.45 ms, thread use time: 147315.90 ms
//        total time: 147317.15



