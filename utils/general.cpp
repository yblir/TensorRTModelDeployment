//
// Created by FH on 2023/1/14.
//
#include <iostream>
#include <dlfcn.h>
#include "general.h"

//记录当前时间点
void Timer::timeStart() {
    startPoint = std::chrono::system_clock::now();
}

//记录当前时间
std::chrono::system_clock::time_point Timer::curTimePoint() {
    return std::chrono::system_clock::now();
}

// 统计当前时间与上一时间点差值
double Timer::timeCount() {
    useTime = std::chrono::system_clock::now() - startPoint;
    return useTime.count() * 1000;
}

//统计当前时间点与之前任一时间点差值
double Timer::timeCountMs(const std::chrono::system_clock::time_point &t1) {
    useTime = std::chrono::system_clock::now() - t1;
    return useTime.count() * 1000;
}

double Timer::timeCountS(const std::chrono::system_clock::time_point &t1) {
    useTime = std::chrono::system_clock::now() - t1;
    return useTime.count();
}

//遍历文件夹,返回图片矩阵vector
void getImageMatFromPath(const std::filesystem::path &imageDirPath, std::vector<cv::Mat> &matVector) {
    // 只遍历当前文件夹下的第一层结构
    std::filesystem::directory_iterator dirPath(imageDirPath);
    for (auto &it: dirPath) {
        std::string suffix = std::filesystem::path(it).extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png")
            matVector.emplace_back(cv::imread(it.path().string()));
    }
}

//遍历文件夹,返回图片矩阵vector
void getImagePath(const std::filesystem::path &imageDirPath, std::vector<std::string> &imagePaths) {
    // 只遍历当前文件夹下的第一层结构
    std::filesystem::directory_iterator dirPath(imageDirPath);
    for (auto &it: dirPath) {
        std::string suffix = std::filesystem::path(it).extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png")
            imagePaths.emplace_back(it.path().string());
    }
}

//bgr转rgb,并归一化
void BGR2RGB(const cv::Mat &image, float *pinInput) {
    int imageArea = image.cols * image.rows;
    unsigned char *pImage = image.data;

    float *channelB = pinInput + imageArea * 0;
    float *channelG = pinInput + imageArea * 1;
    float *channelR = pinInput + imageArea * 2;

    for (int i = 0; i < imageArea; ++i, pImage += 3) {
        // 注意这里的顺序rgb调换了
        *channelR++ = pImage[0] / 255.0f;
        *channelG++ = pImage[1] / 255.0f;
        *channelB++ = pImage[2] / 255.0f;
//        *channelR++ = pImage[0];
//        *channelG++ = pImage[1];
//        *channelB++ = pImage[2];
    }
}

std::string getEnginePath(const BaseParam &param) {
    cudaError_t cudaStatus;
    cudaDeviceProp prop{};
    std::string gpuName, enginePath;
    // 获取显卡信息
    cudaStatus = cudaGetDeviceProperties(&prop, param.gpuId);

    if (cudaSuccess == cudaStatus) {
        gpuName = prop.name;
        // 删掉显卡名称内的空格
        gpuName.erase(std::remove_if(gpuName.begin(), gpuName.end(), ::isspace), gpuName.end());
    } else {
        logError("get device info error.\n");
        return "";
    }

    std::string fpMode = param.mode == Mode::FP32 ? "FP32" : "FP16";

    // 拼接待构建或加载的引擎路径
    enginePath = param.onnxPath.substr(0, param.onnxPath.find_last_of('.')) + "_" + gpuName + "_" + fpMode + ".engine";

    return enginePath;
}

//todo 加载这个函数需要启用base_infer/infer.h中的MakeAlgorithm. 使用pybind11用不到
//todo 它, 先保留, 看以后是否会用到
//Infer *loadDynamicLibrary(const std::string &soPath) {
//    // todo 这里,要判断soPath是不是合法. 并且是文件名时搜索路径
//    printf("load dynamic lib: %s\n", soPath.c_str());
//    auto soHandle = dlopen(soPath.c_str(), RTLD_NOW);
//    if (!soHandle) {
//        printf("open dll error: %s \n", dlerror());
//        return nullptr;
//    }
//    //找到符号Make_Collector的地址
//    void *void_ptr = dlsym(soHandle, "MakeAlgorithm");
//    char *error = dlerror();
//    if (nullptr != error) {
//        printf("dlsym error:%s\n", error);
//        return nullptr;
//    }
//    auto tmp = reinterpret_cast<ptrdiff_t> (void_ptr);
//    auto clct = reinterpret_cast< CreateAlgorithm >(tmp);
//
//    if (nullptr == clct) return nullptr;
//
//    return clct();
//}

const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknown";
    }
}

void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept {
    if (severity <= Severity::kWARNING) {
        if (severity == Severity::kWARNING) {
            printf("\033[33m%s: %s\033[0m\n",
                   severity_string(severity), msg
            );
        } else if (severity <= Severity::kERROR) {
            printf("\033[31m%s: %s\033[0m\n",
                   severity_string(severity), msg
            );
        } else {
            printf("%s: %s\n",
                   severity_string(severity), msg
            );
        }
    }
}

bool check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line) {
    if (cudaSuccess != code) {
        const char *errName = cudaGetErrorName(code);
        const char *errMsg = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, errName, errMsg);
        return false;
    }
    return true;
}

const char *mode_string(Mode type) {
    switch (type) {
        case Mode::FP32:
            return "FP32";
        case Mode::FP16:
            return "FP16";
        default:
            return "UnknowTRTMode";
    }
}

// hsv转bgr 用得到吗?
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
    float h_plane = ((((unsigned int) id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int) id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}