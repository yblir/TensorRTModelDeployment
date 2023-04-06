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

# include "interface/face_interface.h"
#include "file_base.h"

#define HAVE_FACE_RETINA
#define HAVE_FACE_FEATURE

void getImgFromDir(const std::filesystem::path &inputDir, std::vector<std::string> &out) {
    std::filesystem::directory_iterator dirList(inputDir);
    for (auto &it: dirList) {
        std::string suffix = std::filesystem::path(it).extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png" || suffix == ".jfif")
            out.push_back(it.path().string());
    }
}

int main(int argc, char *argv[]) {
    /*
    argc:参数个数
    *argv: 字符数组,记录输入的参数.可执行文件总在0号位,作为一个参数
    */
    // 判断参数个数, 若不为3,终止程序
    if (argc != 3) {
        std::cout << " the number of param is incorrect, must be 3, but now is " << argc << std::endl;
        std::cout << "param format is ./AiSdkDemo [gpu_id] [img_dir_path]" << std::endl;
        return -1;
    }
    FaceTotalConfig conf;

    //初始化引擎
    Handle engine;
    int ret = initEngine(engine, &conf);
    if (ret != 0)
        return ret;

    strcpy(conf.retinaConf.modelFile, "./models/face_yolo_trt/face_detect_v0.5.0_6dca99de68468ca8908e3353dda2b546.onnx");
    strcpy(conf.featureConf.modelFile, "./models/face_feature_trt/face_extract_4.0_ca4e02bff65214a019328c75a3240976.onnx");
    strcpy(conf.poseConf.modelFile, "./models/face_pose_trt/face_pose_v1.4_a77b7d431cf0c22cf60c19267ca187e2.onnx");
    strcpy(conf.qualityConf.model_file, "./models/face_quality_trt/face_extreme_v0.4_e89bcd21685c6b49e90845aba84ba3ae.onnx");
    strcpy(conf.sharpnessConf.modelFile, "./models/face_sharpness_trt/face_sharpness_v0.1_5c32810f754c81d1ce3ea0b403032d54.onnx");

    conf.bitQualityType = 0x3f;     // 这是在干嘛 ?
    conf.deviceId = int(strtol(argv[1], nullptr, 10));
    conf.score_sface_thresh = 0.9f;

    //创建输出文件夹
    std::string path1 = std::string(argv[2]) + "/";
    std::string path2 = path1 + "output/";

    std::filesystem::path imgInputDir(path1);
    std::filesystem::path imgOutputDir(path2);
    //检查文件夹路径是否合法, 检查输出文件夹路径是否存在,不存在则创建
    // 输入不是文件夹,或文件不存在抛出异常
    if (!std::filesystem::exists(imgInputDir) || !std::filesystem::is_directory(imgInputDir))
        return -1;
    //创建输出文件夹
    if (!std::filesystem::exists(imgOutputDir)) std::filesystem::create_directories(imgOutputDir);


    std::cout << "init ok !" << std::endl;

    std::vector<std::string> out;
    getImgFromDir(imgInputDir, out);

    double interTime = 0.0f;
    std::chrono::system_clock::time_point t1;

    for (auto &it: out) {
        cv::Mat img = cv::imread(it);
        //记录当前时间
        t1 = std::chrono::system_clock::now();
        //记录人脸推理结果
        struct FaceResult *resPtr = nullptr;

        int faceNum = 0, minFaceSize = 20, mode = 1;
        ret = inferEngine(engine, img.data, img.cols, img.rows, minFaceSize, mode, FAS_PF_RGB24_B8G8R8, faceNum);
        std::chrono::duration<double> useTime = std::chrono::system_clock::now() - t1;

        interTime += useTime.count();
        if (ret != 0) {
            std::cout << "======== infer failed. use time = " << useTime.count() * 1000 << "ms =========" << std::endl;
            continue;
        }

        if (faceNum > 0)  // ???
            resPtr = new FaceResult[faceNum];
        getResults(engine, faceNum, resPtr);

        //输出推理结果
        for (int i = 0; i < faceNum; ++i) {
            float x1 = resPtr[i].x1, y1 = resPtr[i].y1, x2 = resPtr[i].x2, y2 = resPtr[i].y2;
            float confidence = resPtr[i].confidence;
            float angleP = resPtr[i].angleP, angleR = resPtr[i].angleR, angleY = resPtr[i].angleY;
            float landmark[10];
            for (int j = 0; j < 10; ++j)
                landmark[j] = resPtr[i].landmark[j];

            int qualityType = resPtr[i].qualityType;
            float qualityScore = resPtr[i].qualityScore;

            std::cout << "=============face num " << i + 1 << "==============" << std::endl;
            std::cout << "rect = (" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ")" << std::endl;
            std::cout << "confidence = " << confidence << std::endl;
            std::cout << "angle(p,r,y) = (" << angleP << ", " << angleR << ", " << angleY << ")" << std::endl;
            std::cout << "quality_type = " << qualityType << std::endl;
            std::cout << "quality_score = " << qualityScore << std::endl;

            std::cout << "landmark = ";
            for (float j: landmark) std::cout << j << ", ";

            std::cout << "feature = ";
            for (float j: resPtr[i].feature) std::cout << j << ", ";

            std::cout << std::endl;

            delete[] resPtr, resPtr = nullptr;
        }
        releaseEngine(engine);
    }
    return 0;
}
