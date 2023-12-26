# -*- coding: utf-8 -*-
# @File: thread_py11.py
# @Author: yblir
# @Time: 2023/8/12 10:19
# @Explain: 
# ===========================================
# import tryPybind as ty
import colorsys
import sys
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
        confidence=detection[5]
        color = colors[int(label)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        #cv2.putText(image, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

param = dp.ManualParam()
engine = dp.Engine()

param.fp16 = False
param.gpuId = 0
param.batchSize = 16
param.scoreThresh = 0.5
param.iouThresh = 0.5
param.classNums = 80

param.inputHeight = 640
param.inputWidth = 640

param.onnxPath = '../models/yolov5s.onnx'
param.inputName = 'images'
param.outputName = 'output'

a = engine.initEngine(param)

print('\n==============================')
# img1 = cv2.imread('../imgs/2007_000925.jpg')
# img2 = cv2.imread('../imgs/2007_001311.jpg')
#
# res1 = engine.inferEngine([img1, img2])
# pprint(res1.get())

if __name__ == '__main__':
    colors=set_color()
    root_path = Path(r'/mnt/e/localDatasets/voc/voc_test_100')
    batch_imgs = []
    total_time = 0

    imgs_path=[]
    for img_path in root_path.iterdir():
        if img_path.suffix != '.jpg':
            continue
        img = cv2.imread(str(img_path))
        imgs_path.append(img_path)
        batch_imgs.append(img)
        if len(batch_imgs) < 32:
            continue
        t1 = time.time()
        res = engine.inferEngine(batch_imgs)
        infer_res=res.get()
        t2 = time.time() - t1
        total_time += t2
        for i,img in enumerate(batch_imgs):
            draw_boxes(infer_res[i],img,colors)
            cv2.imwrite(f'/mnt/e/localDatasets/voc/voc_test_100/output/{imgs_path[i].name}',img)
        imgs_path.clear()
        batch_imgs.clear()
    print(total_time)
    # time.sleep(1)
    engine.releaseEngine()
