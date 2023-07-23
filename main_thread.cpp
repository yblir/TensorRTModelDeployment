#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include <dirent.h>

#include "interface/face_interface_thread.h"
//#include "interface/face_interface.h"
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
//    param.yoloDetectParam.gpuId = int(strtol(argv[1], nullptr, 10));
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
//    std::string path1="/mnt/f/LearningData/voc_test_100/";
//    std::string path1 = "/mnt/e/BaiduNetdiskDownload/VOCdevkit/voc_test_10/";
//    std::string path1 = "/mnt/e/BaiduNetdiskDownload/VOCdevkit/voc_test_6000/";
//    std::string path1 = "/mnt/d/Datasets/VOCdevkit/voc_test_300/";
    std::string path1="/mnt/e/localDatasets/voc/voc_test_100/";
    std::filesystem::path imgInputDir(path1);
    std::filesystem::path imgOutputDir(path1 + "output/");

    //检查文件夹路径是否合法, 检查输出文件夹路径是否存在,不存在则创建
    // 输入不是文件夹,或文件不存在抛出异常
    if (!std::filesystem::exists(imgInputDir) || !std::filesystem::is_directory(imgInputDir))
        return -1;
    //创建输出文件夹
    if (!std::filesystem::exists(imgOutputDir))
        std::filesystem::create_directories(imgOutputDir);

    std::map<std::string, batchBoxesType> res;
    std::vector<std::string> imagePaths;
    // 获取该文件夹下所有图片绝对路径,存储在vector向量中
    getImagePath(imgInputDir, imagePaths);

    auto t = timer->curTimePoint();
    std::vector<std::string> batch;
    std::vector<cv::Mat> batchImgs;
    InputData data;
    int count = 0;
    int i = 0;
    double inferTime, total1, hua;
    auto t8 = timer->curTimePoint();
//    int em=0;

    std::map<std::basic_string<char>, std::vector<std::vector<std::vector<float>>>> curResult;
    for (auto &item: imagePaths) {
        batch.emplace_back(item);
        batchImgs.emplace_back(cv::imread(item));
        count += 1;

        if (count >= 5) {
            data.mats = batchImgs;
//            printf("-----------------------------\n");
//            data.imgPath=item;
//            data.mat=cv::imread(item);
//            data.imgPaths=batch;
            auto tt1 = timer->curTimePoint();
            curResult = inferEngine(param, func, data);

            inferTime += timer->timeCount(tt1);
            int j = 0;

            auto yoloRes = curResult["yoloDetect"];
//            printf("5555\n");
            auto tb = timer->curTimePoint();
            for (auto &out: yoloRes) {
                if (out.empty()) {
                    i += 1;
                    j += 1;
                    continue;
                }
                cv::Mat img = cv::imread(batch[j]);
                // 遍历一张图片中每个预测框,并画到图片上
                for (auto &box: out) {
                    drawImage(img, box);
                }
                // 把画好框的图片写入本地
//                std::string drawName = "draw" + std::to_string(i++) + ".jpg";
                cv::imwrite(imgOutputDir / batch[j].substr(batch[j].find_last_of('/') + 1), img);
                j++;
            }
            hua += timer->timeCount(tb);
            batch.clear();
            batchImgs.clear();
            count = 0;
////            break;
        }
    }

    total1 = timer->timeCount(t8);
    printf("right over! %.2f, %.2f,  %.2f\n", inferTime, total1, hua);
    return 0;
}

//right over! 6625.20, 10552.86,  2645.14
//infer use time: 5533.37 ms
//pre   use time: 711.84 ms
//post  use time: 49.46 ms