import time

from fastapi import Form




from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt
import cv2

import torch
import onnxruntime
#import Albumentations as A
import random

from help_funcs import (non_max_suppression, preprocess, mns, plot_result_image, results_to_json, onnx_results_to_json, deepsort_results_to_json)
import numpy as np

from tracker.deep_sort import DeepSort

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_selection_options = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
model_dict = {model_name: None for model_name in model_selection_options}  # set up model cache
classes =['bicycle', 'person', 'stroller', 'wheelchair']
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting


def get_stream_cam(model_name: str = Form(...),
                   img_size: int = Form(640)):
    '''
    Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    # 하드 코딩 된 부분 >> 인자 받아보게 변경, yaml or from html
    mode = 'onnx_deep'
    model_root = './models/'
    #segmentation model
    
    
    config=os.path.join(model_root,'sm_icnet.py')
    seg_ckpt=os.path.join(model_root,'newset_best_mIoU_epoch_82.pth')
    
    if mode == 'onnx':
        model = onnxruntime.InferenceSession(os.path.join(model_root, (model_name + '.onnx')))
    
    elif mode == 'onnx_deep':
        model = onnxruntime.InferenceSession(os.path.join(model_root,'yol5_cls4_best.onnx'))
        ds = DeepSort(os.path.join(model_root, 'ckpt.t7'))
    
    elif mode == 'torch':
        model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(model_root,'yolov5_s_train_hs.pt'))
    
    elif mode == 'torch_deep':
        model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(model_root,'yolov5_s_train_hs.pt'))
        ds = DeepSort(os.path.join(model_root, 'ckpt.t7'))
    else:
        print('not exist mode!! check again!!')
        return

    # Image from Camera no.0
    videos_root = './videos'
    V = os.path.join(videos_root, 'test_2.mp4')
    img_batch = cv2.VideoCapture(V)
    fps = img_batch.get(cv2.CAP_PROP_FPS)
    ret, frame = img_batch.read()
    approx=seg_inference(config,seg_ckpt,frame)
    
  #print("approx확인:",approx)

    total_time_st = time.time_ns()
    print('ori_FPS', fps)

    while img_batch.isOpened():
        ret, frame = img_batch.read()
        if ret:
            if mode == 'onnx':
                img, inf_time = inference_with_onnx(frame, model)
            elif mode == 'onnx_deep':
                img, inf_time = inference_with_onnx_deepsort(frame, model, ds,approx)
            elif mode == 'torch':
                img, inf_time = inference_with_pt(frame, model, img_size)
            elif mode == 'torch_deep':
                img, inf_time = inference_with_pt_deepsort(frame, model, img_size, ds)

            print('FPS', 1 / inf_time * 1000)

            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   img.tobytes() + b'\r\n')

        else:
            print('Fail to read  frame!')
            break
    total_time_end = time.time_ns()
    print('total time :', (total_time_end-total_time_st) / 1000000, '(ms)')
    
    
    
    
    
def seg_inference(config_file,ckpt_model,frame):
    start_ns = time.time_ns()
    frame=cv2.resize(frame,(640,640))
    #성민님 모델
    model = init_segmentor(config_file, ckpt_model, device='cpu')
    result = inference_segmentor(model, frame)
    result[0]=result[0].astype(np.uint8) #int8, int6 타입으로 변경해야 findContours을 사용가능

    contours_approx_simple, _ = cv2.findContours(result[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_approx_simple:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx
        

   
   


def inference_with_onnx(frame, model):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    start_ns = time.time_ns()
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)

    input_data = preprocess(frame)

    input_name = model.get_inputs()[0].name

    # Inference
    results = model.run([], {input_name: input_data})
    nmx_results = non_max_suppression(torch.tensor(results[0]), conf_thres=0.5)[0]  # BATCH SIZE is 1.
    json_results = onnx_results_to_json([nmx_results], classes=classes)

    inf_time = (time.time_ns() - start_ns) / 1000000

    #ret, img = plot_result_image(json_results, frame, colors, inf_time=inf_time)
    ret, img = plot_result_image(json_results, frame, colors)

    return img, inf_time


def inference_with_onnx_deepsort(frame, model, ds,approx):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    start_ns = time.time_ns()

    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
    input_data = preprocess(frame)
    input_name = model.get_inputs()[0].name
    boxes = []

    # Inference
    results = model.run([], {input_name: input_data})
    dets = non_max_suppression(
        torch.tensor(results[0]),
        conf_thres=0.4,
        iou_thres=0.5
    )[0]

    for det in dets:
        det = det.cpu().detach().numpy()
        x1, y1, x2, y2, score, class_num = det.astype(np.float32)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        boxes.append([cx, cy, w, h, class_num])

    tracked_boxes = ds.update(boxes, frame)

    json_results = deepsort_results_to_json([tracked_boxes], classes=classes)

    inf_time = (time.time_ns() - start_ns) / 1000000

    #ret, img = plot_result_image(json_results, frame, colors, inf_time=inf_time, deepsort=True)
    ret, img = plot_result_image(json_results, frame, colors, approx, deepsort=True)

    return img, inf_time


def inference_with_pt(frame, model, img_size):
    start_ns = time.time_ns()

    results = model(frame.copy(), size=img_size)
    json_results = results_to_json(results, model)

    inf_time = (time.time_ns() - start_ns) / 1000000

    #ret, img = plot_result_image(json_results, frame, colors, inf_time=inf_time)
    ret, img = plot_result_image(json_results, frame, colors)

    return img, inf_time


def inference_with_pt_deepsort(frame, model, img_size, ds):
    start_ns = time.time_ns()
    boxes = []
    results = model(frame.copy(), size=img_size)

    for result in results.xyxy:
        for det in result:
            det = det.cpu().detach().numpy()
            x1, y1, x2, y2, score, class_num = det.astype(np.float32)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1
            boxes.append([cx, cy, w, h, class_num])

    tracked_boxes = ds.update(boxes, frame)

    json_results = deepsort_results_to_json([tracked_boxes], classes=classes)

    inf_time = (time.time_ns() - start_ns) / 1000000

    #ret, img = plot_result_image(json_results, frame, colors, inf_time=inf_time, deepsort=True)
    ret, img = plot_result_image(json_results, frame, colors, deepsort=True)

    return img, inf_time


