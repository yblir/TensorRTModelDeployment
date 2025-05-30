#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include <dirent.h>

//#include "interface/interface_thread.h"
#include "interface/single_interface.h"
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
        std::cout << " the number of engine is incorrect, must be 3, but now is " << argc << std::endl;
        std::cout << "engine format is ./AiSdkDemo gpu_id img_dir_path" << std::endl;
        return -1;
    }

    // =====================================================================
    struct ManualParam inputParam;

//    inputParam.onnxPath = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
//    inputParam.enginePath = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/yolov5s_NVIDIAGeForceGTX1080_FP32.engine";
    inputParam.onnxPath="/mnt/e/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
    inputParam.gpuId = 0;
    inputParam.batchSize = 16;
    inputParam.inputHeight = 640;
    inputParam.inputWidth = 640;

    inputParam.inputName = "images";
    inputParam.outputName = "output";

    inputParam.iouThresh = 0.5;
    inputParam.scoreThresh = 0.5;

    auto engine = Engine();
    int ret = engine.initEngine(inputParam);
    if (ret != 0)
        return ret;
    std::cout << "init ok !" << std::endl;

    // ============================================================================================
//    std::string path1 = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/imgs";
     std::string path1 = "/mnt/e/localDatasets/voc/voc_test_100/";

    std::filesystem::path imgInputDir(path1);
    std::filesystem::path imgOutputDir(path1 + "output/");

    //检查文件夹路径是否合法, 检查输出文件夹路径是否存在,不存在则创建
    // 输入不是文件夹,或文件不存在抛出异常
    if (!std::filesystem::exists(imgInputDir) || !std::filesystem::is_directory(imgInputDir)) {
        logError("imgInputDir does not exist");
        return -1;
    }
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
//    InputData data;
    int count = 0;

    double inferTime, total1, hua;
    auto t8 = timer->curTimePoint();
//    int em=0;

    auto lastElement = &imagePaths.back();
//    std::map<std::basic_string<char>, std::vector<std::vector<std::vector<float>>>> curResult;
    for (auto &item: imagePaths) {
        batch.emplace_back(item);
        batchImgs.emplace_back(cv::imread(item));
        count += 1;

        if (count >= 32 or &item == lastElement) {
//            data.mats = batchImgs;
            auto tt1 = timer->curTimePoint();

            auto yoloRes = engine.inferEngine(batchImgs);
            inferTime += timer->timeCountS(tt1);
            int j = 0;

            auto tb = timer->curTimePoint();
            for (auto &out: yoloRes) {
                if (out.empty()) {
                    j += 1;
                    continue;
                }
                cv::Mat img = cv::imread(batch[j]);
                // 遍历一张图片中每个预测框,并画到图片上
                for (auto &box: out) {
                    drawImage(img, box);
                }
                // 把画好框的图片写入本地
                cv::imwrite(imgOutputDir / batch[j].substr(batch[j].find_last_of('/') + 1), img);
                j++;
            }
            hua += timer->timeCountS(tb);
            batch.clear();
            batchImgs.clear();
            count = 0;
////            break;
        }
    }

    total1 = timer->timeCountS(t8);
    printf("right over! %.3f s, %.3f s,  %.3f s\n", inferTime, total1, hua);
    engine.releaseEngine();
    return 0;
}


//right over! 0.004 s, 11.560 s,  6.411 s
//2023-10-21 17:15:51   thread_infer.cpp:398  INFO| start executing destructor ...
//2023-10-21 17:15:51   thread_infer.cpp:269  INFO| infer use time: 1.863 s
//2023-10-21 17:15:51   thread_infer.cpp:211  INFO| pre   use time: 0.893 s
//2023-10-21 17:15:51   thread_infer.cpp:328  INFO| post  use time: 0.117 s
//2023-10-21 17:15:51interface_thread.cp:122  SUCC| Release engine success

//单线程多线程预处理
//right over! 0.570 s, 4.735 s,  2.627 s
//right over! 0.565 s, 4.992 s,  2.777 s
//right over! 0.568 s, 5.074 s,  2.828 s

// 单线程单线线程预处理
//right over! 1.587 s, 6.083 s,  2.857 s
//right over! 1.593 s, 6.153 s,  2.902 s
//right over! 1.592 s, 6.034 s,  2.843 s

//多线程单线程预处理
//right over! 1.418 s, 5.925 s,  2.831 s
//right over! 1.413 s, 5.931 s,  2.860 s

// 多线程多线程预处理和单线程多线程预处理推理总耗时差不多, 快0.1s.
//right over! 0.467 s, 4.864 s,  2.786 s
//right over! 0.476 s, 4.903 s,  2.866 s
//right over! 0.464 s, 4.965 s,  2.827 s