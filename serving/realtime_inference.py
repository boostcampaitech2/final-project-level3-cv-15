import time

from fastapi import Form

import cv2

import torch
import onnxruntime

import random

from help_funcs import (non_max_suppression, preprocess, mns, plot_result_image, results_to_json, onnx_results_to_json)
import numpy as np

from tracker.deep_sort import DeepSort

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_selection_options = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
model_dict = {model_name: None for model_name in model_selection_options}  # set up model cache
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch','potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard','cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush']
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

    if mode == 'torch':
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    elif mode == 'torch_deep':
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        ds = DeepSort(os.path.join(model_root, 'ckpt.t7'))
    elif mode == 'onnx':
        model = onnxruntime.InferenceSession(os.path.join(model_root, (model_name + '.onnx')))
    elif mode == 'onnx_deep':
        model = onnxruntime.InferenceSession(os.path.join(model_root, (model_name + '.onnx')))
        ds = DeepSort(os.path.join(model_root, 'ckpt.t7'))

    # Image from Camera no.0
    videos_root = './videos'
    V = os.path.join(videos_root, 'test_2.mp4')
    img_batch = cv2.VideoCapture(V)
    fps = img_batch.get(cv2.CAP_PROP_FPS)

    total_time_st = time.time_ns()

    while img_batch.isOpened():
        ret, frame = img_batch.read()
        if ret:
            start_ns = time.time_ns()
            if mode == 'onnx':
                img = inference_with_onnx(frame, model)
            elif mode == 'onnx_deep':
                img = inference_with_onnx_deepsort(frame, model, ds)
            elif mode == 'torch':
                img = inference_with_pt(frame, model, img_size)
            elif mode == 'torch_deep':
                img = inference_with_pt(frame, model, img_size)

            # time check end
            end_ns = time.time_ns()
            print('time :', (end_ns-start_ns) / 1000000, '(ms)')
            print('FPS', 1 / (end_ns-start_ns) * 1000000000)
            print('ori_FPS', fps)

            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   img.tobytes() + b'\r\n')

            # 100ms 지연
            #if cv2.waitKey(100) > 0:
            #    break
        else:
            print('Fail to read  frame!')
            break
    total_time_end = time.time_ns()
    print('total time :', (total_time_end-total_time_st) / 1000000, '(ms)')


def inference_with_onnx(frame, model):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)

    input_data = preprocess(frame)

    input_name = model.get_inputs()[0].name

    # Inference
    results = model.run([], {input_name: input_data})
    nmx_results = non_max_suppression(results[0], conf_thres=0.5)[0]  # BATCH SIZE is 1.
    json_results = onnx_results_to_json([nmx_results], classes=classes)

    ret, img = plot_result_image(json_results, frame, colors)

    return img


def inference_with_onnx_deepsort(frame, model, ds):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
    input_data = preprocess(frame)
    input_name = model.get_inputs()[0].name
    boxes = []

    # Inference
    results = model.run([], {input_name: input_data})
    dets = mns(
        torch.tensor(results[0]),
        conf_thres=0.4,
        iou_thres=0.5
    )[0]

    for det in dets:
        det = det.cpu().detach().numpy()
        x1, y1, x2, y2, score = det.astype(np.float32)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        boxes.append([cx, cy, w, h])

    tracked_boxes = ds.update(boxes, frame)

    # class 작업되면 plot_result_image로 바꾸기
    for bbox in tracked_boxes:
        x, y, w, h, trk_id = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, str(trk_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    ret, img = cv2.imencode('.jpg', frame)

    return img


def inference_with_pt(frame, model, img_size):
    results = model(frame.copy(), size=img_size)
    json_results = results_to_json(results, model)

    ret, img = plot_result_image(json_results, frame, colors)

    return img


def inference_with_pt_deepsort(frame, model, img_size):
    results = model(frame.copy(), size=img_size)
    json_results = results_to_json(results, model)

    ret, img = plot_result_image(json_results, frame)

    return img
