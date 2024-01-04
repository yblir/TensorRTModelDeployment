# -*- coding: utf-8 -*-
# @File: thread_py11.py
# @Author: yblir
# @Time: 2023/8/12 10:19
# @Explain: 
# ===========================================
# import tryPybind as ty
import colorsys
import sys
import os
import time
import cv2
from PIL import Image

from pprint import pprint
import numpy as np
from ctypes import cdll
from pathlib2 import Path

cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer.so')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_builder_resource.so.8.6.1')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so')
cdll.LoadLibrary('/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvonnxparser.so')
sys.path.append('../cmake-build-debug')
# sys.path.append('../utils')
import deployment as dp


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


def set_color():
    '''
    设置绘制的边框颜色
    :return:
    '''
    # 画框设置不同的颜色
    hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
    # *x: 解包(10,1.,1,.)这样的结构
    color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # [(12,233,9),(...),(...)]  # 每个小元组就是一个rgb色彩值
    color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
    # 打乱颜色
    np.random.seed(10101)
    np.random.shuffle(color)
    return color


def draw_boxes(detections, image, colors):
    for detection in detections:
        if not detection:
            continue
        x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
        label = detection[4]
        confidence = detection[5]
        color = colors[int(label)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        # cv2.putText(image, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


param = dp.ManualParam()
engine = dp.Engine()
# todo fp16有时即使编译成功,也会推理时发生segment default错误. 原因未知! 推测与可能与onnx文件有关, 难道要改变转onnx时参数的精度?
param.fp32 = True
# param.fp16 = True

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

init_flag = engine.initEngine(param)
if init_flag == -1:
    # print("init error")
    raise ValueError("init error")

if __name__ == '__main__':
    colors = set_color()
    root_path = Path(r'/mnt/e/localDatasets/voc/voc_test_1000')
    batch_imgs = []
    total_time = 0

    last_path = [i for i in root_path.iterdir() if i.suffix == '.jpg'][-1]

    os.makedirs("/mnt/e/localDatasets/voc/voc_test_1000/output", exist_ok=True)

    total_time = 0
    imgs_path = []
    for k, img_path in enumerate(root_path.iterdir()):
        if img_path.suffix != '.jpg':
            continue
        img = cv2.imread(str(img_path))
        # img=img.astype(np.float16)
        # print(type(img))
        imgs_path.append(img_path)
        batch_imgs.append(img)

        if len(batch_imgs) < 32 and img_path != last_path:
            continue
        t1 = time.time()
        res = engine.inferEngine(batch_imgs)
        # res = engine.inferEngine(img)
        # print('1111')
        # infer_res = res.get()
        infer_res = res
        t2 = time.time() - t1
        total_time += t2

        # draw_boxes(infer_res[0], img, colors)
        # print(infer_res[0])
        # cv2.imwrite(f'/mnt/e/localDatasets/voc/voc_test_1000/output/{img_path.name}', img)

        for i, img in enumerate(batch_imgs):
            if not infer_res[i]:
                continue
            draw_boxes(infer_res[i], img, colors)
            cv2.imwrite(f'/mnt/e/localDatasets/voc/voc_test_1000/output/{imgs_path[i].name}', img)

        imgs_path.clear()
        batch_imgs.clear()

        print(total_time)
        # time.sleep(1)
        # print('==============================\n')
    engine.releaseEngine()

# 2.09, 2.15 2.06 2.17 2.08
