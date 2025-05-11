#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include <dirent.h>

#include "interface/thread_interface.h"
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
    if (argc != 3) {
        std::cout << " the number of engine is incorrect, must be 3, but now is " << argc << std::endl;
        std::cout << "engine format is ./AiSdkDemo gpu_id img_dir_path" << std::endl;
        return -1;
    }

    // =====================================================================
    struct ManualParam inputParam;

    inputParam.onnxPath = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/models/yolov5s.onnx";
    inputParam.enginePath = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/yolov5s_NVIDIAGeForceGTX1080_FP32.engine";
    inputParam.gpuId = 0;
    inputParam.batchSize = 5;
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
     std::string path1 = "/media/xk/D6B8A862B8A8433B/GitHub/TensorRTModelDeployment/imgs";
//    std::string path1 = "/mnt/e/localDatasets/voc/voc_test_1000/";

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
    InputData data;
    int count = 0,aa=0;

    double inferTime, total1, hua;
    auto t8 = timer->curTimePoint();
//    int em=0;
    auto lastElement = &imagePaths.back();
//    std::map<std::basic_string<char>, std::vector<std::vector<std::vector<float>>>> curResult;
    for (int i = 0; i < 2; ++i) {

        for (auto &item: imagePaths) {
            batch.emplace_back(item);
            batchImgs.emplace_back(cv::imread(item));
            count += 1;

            if (count >= 10 or &item == lastElement) {
                data.mats = batchImgs;
                auto tt1 = timer->curTimePoint();

                auto futureRes = engine.inferEngine(data.mats);

                int j = 0;
//            auto yoloRes = curResult["yoloDetect"];
                auto yoloRes = futureRes.get();
                inferTime += timer->timeCountS(tt1);
                auto tb = timer->curTimePoint();
                for (auto &out: yoloRes) {
                    if (out.empty()) {
                        j += 1;
                        continue;
                    }
                    cv::Mat img = cv::imread(batch[j]);
                    aa += 1;
                    // 遍历一张图片中每个预测框,并画到图片上
                    for (auto &box: out) {
//                    std::cout<<"box="<<box[0]<<", "<<box[1]<<", "<<box[2]<<", "<<box[3]<<", "<<box[4]<<", "<<box[5]<<std::endl;
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
    }

    total1 = timer->timeCountS(t8);
    printf("right over! %.3f s, %.3f s,  %.3f s\n", inferTime, total1, hua);
    std::cout<<"aa="<<aa<<std::endl;
    engine.releaseEngine();
    return 0;
}


//right over! 0.004 s, 11.560 s,  6.411 s
//2023-10-21 17:15:51   thread_infer.cpp:398  INFO| start executing destructor ...
//2023-10-21 17:15:51   thread_infer.cpp:269  INFO| infer use time: 1.863 s
//2023-10-21 17:15:51   thread_infer.cpp:211  INFO| pre   use time: 0.893 s
//2023-10-21 17:15:51   thread_infer.cpp:328  INFO| post  use time: 0.117 s
//2023-10-21 17:15:51interface_thread.cp:122  SUCC| Release engine success

//right over! 0.001 s, 2.489 s,  1.308 s
//2023-10-22 19:52:58   thread_infer.cpp:398  INFO| start executing destructor ...
//2023-10-22 19:52:58   thread_infer.cpp:211  INFO| pre   use time: 0.232 s
//2023-10-22 19:52:58   thread_infer.cpp:269  INFO| infer use time: 0.115 s
//2023-10-22 19:52:58   thread_infer.cpp:328  INFO| post  use time: 0.018 s

//right over! 0.546 s, 1.936 s,  0.927 s
//right over! 0.545 s, 1.992 s,  0.968 s
//right over! 0.541 s, 1.973 s,  0.954 s
//right over! 0.553 s, 1.956 s,  0.938 s
//right over! 0.548 s, 1.996 s,  0.968 s

//single
//right over! 0.581 s, 5.143 s,  2.877 s
//right over! 0.572 s, 5.336 s,  3.069 s
//right over! 0.575 s, 5.142 s,  2.866 s
//right over! 0.570 s, 5.070 s,  2.830 s
//right over! 0.580 s, 4.880 s,  2.698 s

//right over! 0.543 s, 5.105 s,  2.858 s

//right over! 0.585 s, 5.081 s,  2.837 s
//right over! 0.599 s, 4.793 s,  2.641 s
//right over! 0.582 s, 5.148 s,  2.861 s


//right over! 0.600 s, 5.216 s,  2.915 s