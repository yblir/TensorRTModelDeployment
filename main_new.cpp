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

#include "interface/face_interface_new.h"
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

    param.yoloDetectParam.onnxPath = "/mnt/i/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
//    param.yoloDetectParam.onnxPath = "/mnt/e/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
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
    // =====================================================================

    //创建输出文件夹
    std::string path1 = std::string(argv[2]) + "/";

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

    inferEngine(param, func, imagePaths, outs);

    std::cout << "在原图上画框" << std::endl;
    int i = 0;
    // 画yolo目标检测框
    if (!outs.detectResult.empty()) {
        // 遍历每张图片
        for (auto &out: outs.detectResult) {
            cv::Mat img = cv::imread(imagePaths[i]);
            // 遍历一张图片中每个预测框,并画到图片上
            for (auto &box: out) {
                drawImage(img, box);
            }
            // 把画好框的图片写入本地
            std::string drawName = "draw" + std::to_string(i) + ".jpg";
            cv::imwrite(drawName, img);
            i += 1;
        }
    }
//    printf(reinterpret_cast<const char *>('fsdfsdfsfd\n'));
    return 0;
}

