import time

from fastapi import Form

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from tracker.deep_sort import DeepSort

import cv2
import torch
import onnxruntime
import random
import numpy as np

from help_funcs import (non_max_suppression, preprocess, plot_result_image, results_to_json)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_selection_options = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
model_dict = {model_name: None for model_name in model_selection_options}  # set up model cache
classes =['bicycle', 'person', 'stroller', 'wheelchair']
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting


def get_stream_cam(model_name: str = Form(...),
                   img_size: int = Form(640),
                   mode: str = Form(...)
                   ):
    '''
    Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    model_root = './models/'

    # segmentation model
    config = os.path.join(model_root, 'hyuns_bise.py')
    seg_ckpt = os.path.join(model_root, 'biseNet_best_mIoU2_epoch_268.pth')

    if 'onnx' in mode:
        model = onnxruntime.InferenceSession(os.path.join(model_root, (model_name + '.onnx')))
    elif 'torch' in mode:
        model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(model_root, (model_name+'.pt')))
    else:
        print('not exist model!! check again!!')
        return

    if 'deep' in mode:
        ds = DeepSort(os.path.join(model_root, 'ckpt.t7'))

    # Image from Camera no.0
    videos_root = './videos'
    V = os.path.join(videos_root, 'test_1.mp4')
    img_batch = cv2.VideoCapture(V)

    fps = img_batch.get(cv2.CAP_PROP_FPS)
    
    # segmentation inference
    ret, frame = img_batch.read()
    approx, seg_time = seg_inference(config, seg_ckpt, frame)

    if approx == []:
        print('감지된 횡단 보도가 없습니다! 카메라를 재조정해주세요!')
        return

    #print("approx확인:",approx)
    #print("segtime:", seg_time)

    total_time_st = time.time_ns()
    print('ori_FPS', fps)

    while img_batch.isOpened():
        ret, frame = img_batch.read()

        if ret:
            if 'deep' in mode:
                img, inf_time = inference(frame, img_size, model, approx, mode, ds=ds)
            else:
                img, inf_time = inference(frame, img_size, model, approx, mode)

            print('FPS', 1 / inf_time * 1000)

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
    
    
# segmentation polygon 좌표 function
def seg_inference(config_file, ckpt_model, frame):
    start_ns = time.time_ns()

    select = [0]
    approx_list = [0]
    approx = []

    frame = cv2.resize(frame, (640,640))

    # 성민님 모델
    model = init_segmentor(config_file, ckpt_model, device='cpu')
    result = inference_segmentor(model, frame)
    result = result[0].astype(np.uint8) #int8, int6 타입으로 변경해야 findContours을 사용가능

    contours_approx_simple, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_approx_simple:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        a = cv2.contourArea(approx)

        if a > select[0]:
            select.pop()
            approx_list.pop()
            select.append(a)
            approx_list.append([approx])
        else:
            continue

    inf_time = (time.time_ns() - start_ns) / 1000000

    return approx, inf_time


def inference(frame, img_size, model, approx, mode, ds=None):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)

    start_ns = time.time_ns()

    # Object Detection
    if 'onnx' in mode:
        input_data = preprocess(frame)

        # Inference
        input_name = model.get_inputs()[0].name
        result_onnx = model.run([], {input_name: input_data})
        results = non_max_suppression(torch.tensor(result_onnx[0]), conf_thres=0.4, iou_thres=0.5)[0]
    elif 'torch' in mode:
        results_torch = model(frame.copy(), size=img_size)
        results = results_torch.xyxy[0]

    # Object Tracking with Deep Sort
    if 'deep' in mode:
        results = deepsort(frame, results, ds)

    '''
        TODO:
        valid check should be here

        after valid check => pls make boolean: keep_green
    '''

    json_results = results_to_json([results], classes, mode)

    inf_time = (time.time_ns() - start_ns) / 1000000
    ret, img = plot_result_image(json_results, frame, colors, approx, mode)

    return img, inf_time


def deepsort(frame, results, ds):
    boxes = []
    for det in results:
        det = det.cpu().detach().numpy()
        x1, y1, x2, y2, score, class_num = det.astype(np.float32)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = x2 - x1
        h = y2 - y1
        boxes.append([cx, cy, w, h, class_num])

    tracked_boxes = ds.update(boxes, frame)

    return tracked_boxes
