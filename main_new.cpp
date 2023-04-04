#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include <dirent.h>

#include "interface/face_interface_thread.h"

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
//    std::string path1="/mnt/e/cartoon_data/personai_icartoonface_detval/";
//    std::string path1 = "/mnt/e/BaiduNetdiskDownload/VOCdevkit/voc_test_10/";
    std::string path1 = "/mnt/e/BaiduNetdiskDownload/VOCdevkit/voc_test_6000/";
//    std::string path1 = "/mnt/d/VOCdevkit/voc_test/";
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
//    int em=0;
    std::map<std::basic_string<char>, std::vector<std::vector<std::vector<float>>>> curResult;
    for (auto &item: imagePaths) {
        batch.emplace_back(item);
        batchImgs.emplace_back(cv::imread(item));
        count += 1;

        if (count >= 6) {
            data.mats = batchImgs;
//            data.imgPath=item;
//            data.mat=cv::imread(item);
//            data.imgPaths=batch;
            curResult = inferEngine(param, func, data);

            int j = 0;
            auto yoloRes = curResult["yoloDetect"];
//            printf("5555\n");
            for (auto &out: yoloRes) {
                if (out.empty()) {
                    i += 1;
                    j += 1;
                    continue;
                }
//                printf("11111\n");
                cv::Mat img = cv::imread(batch[j]);
//                cv::Mat img= batchImgs[i];
                j += 1;
//                printf("2222\n");
                // 遍历一张图片中每个预测框,并画到图片上
                for (auto &box: out) {
                    drawImage(img, box);
                }
                // 把画好框的图片写入本地
                std::string drawName = "draw" + std::to_string(i) + ".jpg";
                cv::imwrite(imgOutputDir / drawName, img);
                i += 1;

            }
            batch.clear();
            batchImgs.clear();
            count = 0;
//            break;
        }

//        break;
//        printf("em = %d\n",em);
//        std::cout << "OKkkkkkkkkk!" << std::endl;

    }
//    inferEngine(param, func, imagePaths, outs);
//    total = timer->timeCount(t);
//    printf("total time: %.2f\n", total);
//    std::cout << "在原图上画框" << std::endl;
//    int i = 0;
    // 画yolo目标检测框
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
    printf("right over!\n");
    return 0;
}
/*
 for(ff 图片)
   vector(存储有一个batch的结果) = inferengine( 图片路径)
*/
//pre   use time: 14029.04 ms, thread use time: 117828.75 ms, pre img num = 6000
//infer use time: 30711.35 ms, thread use time: 117840.00 ms
//post  use time: 1704.45 ms, thread use time: 117844.82 ms
//total time: 117846.24

//pre   use time: 13982.90 ms, thread use time: 93028.23 ms, pre img num = 6000
//infer use time: 26992.98 ms, thread use time: 93037.26 ms
//post  use time: 1751.12 ms, thread use time: 93041.80 ms
//total time: 93043.74

//pre   use time: 14001.84 ms, thread use time: 94202.40 ms, pre img num = 6000
//infer use time: 27069.24 ms, thread use time: 94211.42 ms
//post  use time: 1768.55 ms, thread use time: 94216.24 ms
//total time: 94217.77

//===============================================================
//pre   use time: 13999.50 ms, thread use time: 96727.27 ms, pre img num = 6000
//infer use time: 27080.28 ms, thread use time: 96736.42 ms
//post  use time: 1779.62 ms, thread use time: 96741.24 ms
//total time: 96742.49

//pre   use time: 13953.80 ms, thread use time: 97768.33 ms, pre img num = 6000
//infer use time: 27350.60 ms, thread use time: 97778.11 ms
//post  use time: 1771.26 ms, thread use time: 97782.84 ms
//total time: 97784.16


// pre   use time: 14056.91 ms, thread use time: 90496.34 ms, pre img num = 6000
//infer use time: 26742.28 ms, thread use time: 90505.30 ms
//post  use time: 1752.58 ms, thread use time: 90510.30 ms
//total time: 90511.79

//infer use time: 32412.12 ms, thread use time: 111476.13 ms
//post  use time: 2455.01 ms, thread use time: 111476.13 ms
//pre   use time: 14393.79 ms, thread use time: 111476.17 ms

//pre   use time: 14457.16 ms, thread use time: 111777.15 ms
//post  use time: 2533.36 ms, thread use time: 111777.14 ms
//infer use time: 32659.31 ms, thread use time: 111777.16 ms

//infer use time: 34023.43 ms, thread use time: 110131.58 ms
//pre   use time: 14514.60 ms, thread use time: 110131.58 ms
//post  use time: 2556.52 ms, thread use time: 110131.52 ms

//pre   use time: 14836.47 ms, thread use time: 190120.89 ms
//infer use time: 30508.79 ms, thread use time: 190120.86 ms
//post  use time: 2219.18 ms, thread use time: 190120.84 ms