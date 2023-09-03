## TensorRTModelDeployment
模型部署多了,很想写一份属于自己的部署代码

![tensorRT.png](imgs%2FtensorRT.png)
- 使用方法
```bash
cd TensorRTModelDeployment
mkdir build
# 根据自己cuda和tensorRT版本修改
cmake ..
make
# 然后python调用生成的部署动态库,例如deployment.cpython-38-x86_64-linux-gnu.so
```

- 测试python调用
```python
cd test
python py_11.py
```
```python
使用python调用打印的log日志:\
(base) root@wsl:/mnt/e/PyCharm/temp_project/test1/py11# python py_11.py
2023-09-03 10:11:06    trt_builder.cpp:114  INFO| engine file is not exist, build engine from onnx file ...
2023-09-03 10:11:06      trt_infer.cpp:85   SUCC| load engine success: yolov5s_NVIDIAGeForceRTX4090_FP16.engine
2023-09-03 10:11:07      trt_infer.cpp:104  SUCC| deserialize cuda engine success
warning: CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
2023-09-03 10:11:07      trt_infer.cpp:113  SUCC| create context success
2023-09-03 10:11:07      trt_infer.cpp:363  SUCC| thread start success !

==============================
[[[309.01141357421875, 105.17643737792969, 458.7785949707031, 336.1158142089844, 18.0, 0.9375417828559875], [36.052734375, 99.38946533203125, 213.8197784423828, 343.53826904296875, 18.0, 0.8890634179115295]]]
```