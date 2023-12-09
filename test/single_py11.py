# -*- coding: utf-8 -*-
# @File: thread_py11.py
# @Author: yblir
# @Time: 2023/8/12 10:19
# @Explain: 
# ===========================================
# import tryPybind as ty

import sys
import time
import cv2
from PIL import Image

from pprint import pprint
import numpy as np
from ctypes import cdll

cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer.so')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_builder_resource.so.8.6.1')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvonnxparser.so')

sys.path.append('../cmake-build-debug')
import deployment as dp

param = dp.ManualParam()
engine = dp.Engine()

param.fp16 = False
param.gpuId = 0
param.batchSize = 1
param.scoreThresh = 0.5
param.iouThresh = 0.5
param.classNums = 80

param.inputHeight = 640
param.inputWidth = 640

param.onnxPath = '../models/yolov5s.onnx'
param.inputName = 'images'
param.outputName = 'output'

a = engine.initEngine(param)


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


print('\n==============================')
img1 = cv2.imread('../imgs/2007_000925.jpg')
img2 = cv2.imread('../imgs/2007_001311.jpg')

# img1 = img1.astype("float32")
# img1 = cv2.resize(img1, (640, 640))
# img1 = np.ascontiguousarray(img1.transpose(2, 0, 1))
# print(img1.shape)
# img2=cv2.resize(img2,(640,640))

# img2 = np.ascontiguousarray(img2.transpose(2, 0, 1))
# print(img2.shape)
# img2=img2.astype('float32')
# img1 = Image.open('../imgs/2007_000925.jpg')
# img1 = letterbox_image(img1, (640, 640))
# img1 = np.asarray(img1)
# img1 = img1.astype("float32") / 255.
# img1 = np.ascontiguousarray(img1.transpose(2, 0, 1))
res = engine.inferEngine([img1,img2])
pprint(res)
# print('\n==============================')
# res2 = engine.inferEngine(img2)
# print(res2.get())

# time.sleep(1)
engine.releaseEngine()
