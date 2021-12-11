import time

import torchvision
from torchvision.ops import box_iou
from fastapi import FastAPI, Form, File, UploadFile

import cv2

import torch
import onnxruntime

import random

import numpy as np
from PIL import Image

import torchvision.transforms as transforms

import base64

import imageio


model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch','potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard','cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush']

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting


def get_stream_cam(model_name: str = Form(...),
                   img_size: int = Form(640)):
    '''
    Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    # 하드 코딩 된 부분 >> 인자 받아보게 변경, yaml or from html
    mode = 'torch'

    if mode == 'torch':
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    elif mode == 'onnx':
        model = onnxruntime.InferenceSession(model_name+'.onnx')

    # Image from Camera no.0
    V='/Users/ansojung/final-project-level3-cv-15/serving/test_1.mp4'
    img_batch = cv2.VideoCapture(V)
    
    
    # FPS 추가 필요
    while True:
        # time check should be here
        if mode == 'onnx':
            img = inference_with_onnx(img_batch, model)
        elif mode == 'torch':
            img = inference_with_pt(img_batch, model, img_size)
        # time check end
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' +
               img.tobytes() + b'\r\n')

        if cv2.waitKey(100) > 0:
            break


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference_with_onnx(img_batch, model):
    # CV2 IMAGE / frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    ret, frame = img_batch.read()

    '''
    # JPG to >> 1 3 640 640
    frame_jpg = frame.astype(np.float32)
    frame_jpg = cv2.resize(frame_jpg, (640, 640), interpolation=cv2.INTER_CUBIC)
    frame_jpg = np.transpose(frame_jpg, axes=(2, 0, 1))
    frame_jpg = np.expand_dims(frame_jpg, axis=0)  # unsqueeze(0)
    '''

    # PIL IMAGE / YCBCR >> jpg to YCbCr(1, 3, 640, 640)
    frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_ycbcr = cv2.resize(frame_ycbcr, (640, 640), interpolation=cv2.INTER_CUBIC)
    frame_ycbcr = Image.fromarray(frame_ycbcr)
    frame_ycbcr = frame_ycbcr.convert('YCbCr')

    to_tensor = transforms.ToTensor()

    frame_ycbcr = to_tensor(frame_ycbcr)
    frame_ycbcr.unsqueeze_(0)

    input_data = to_numpy(frame_ycbcr)
    input_name = model.get_inputs()[0].name

    # Inference
    results = model.run([], {input_name: input_data})
    nmx_results = non_max_suppression(results[0], conf_thres=0.5)[0]  # BATCH SIZE is 1.
    json_results = onnx_results_to_json([nmx_results])

    ret, img = plot_result_image(json_results, frame)

    return img


def inference_with_pt(img_batch, model, img_size):
    ret, frame = img_batch.read()
    results = model(frame.copy(), size=img_size)
    json_results = results_to_json(results, model)

    ret, img = plot_result_image(json_results, frame)

    return img


# with pytorch
def results_to_json(results, model):
    # Converts yolo model output to json (list of list of dicts)
    
    
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],  # convert bbox results to int from float
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxy
    ]


def onnx_results_to_json(results):
    # Converts yolo model output to json (list of list of dicts)
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": classes[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],  # convert bbox results to int from float
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results
    ]


def plot_result_image(json_results, img):
    for bbox_list in json_results:
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label, color=colors[int(bbox['class'])], line_thickness=3)

    return cv2.imencode('.jpg', img)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.6, classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Number of classes.
    nc = prediction[0].shape[1] - 5
    # Candidates.
    xc = prediction[..., 4] > conf_thres

    # Settings:
    # Minimum and maximum box width and height in pixels.
    min_wh, max_wh = 2, 4096

    # Maximum number of detections per image.
    max_det = 300

    #  Timeout.
    time_limit = 10.0

    # Require redundant detections.
    redundant = True

    # Multiple labels per box (adds 0.5ms/img).
    multi_label = nc > 1

    # Use Merge-NMS.
    merge = False

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints:
        # Confidence.
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling.
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image.
        if not x.shape[0]:
            continue

        # Compute conf.
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2).
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls).
        if multi_label:
            i, j = torch.nonzero(torch.Tensor(x[:, 5:] > conf_thres), as_tuple=False).T
            
            # print(i.size(), j.size())
            # i=i.reshape(1,i.size()[0])
            # j=j.reshape(1,j.size()[0])
            # print("i값:",i, "j값",j)
            tmp1=torch.Tensor(box[i])
            tmp2=torch.Tensor(x[i, j + 5, None])
            tmp3=j[:, None].float()
            print(i.size(), j.size())
            if i.size()[0]==1 and j.size()[0]==1:
                print("통과!")
                tmp1=tmp1.reshape(1,tmp1.size()[0])
                tmp2=tmp2.reshape(1,tmp2.size()[0])
    
            # tmp1=torch.Tensor(box[i])
            # tmp2=torch.Tensor(x[i, j + 5, None])
            # tmp3=j[:, None].float()
            
            print('tmp1:',tmp1.size())
            print('tmp2:',tmp2.size())
            print('tmp3:',tmp3.size())
            print('concat:',(tmp1,tmp2,tmp3))
            
            #print(box)
            x=torch.cat((tmp1,tmp2,tmp3),1)
    
            #x = torch.cat((torch.Tensor(box[i]), torch.Tensor(x[i, j + 5, None]), j[:, None].float()), 1)
              
        else:
            # Best class only.
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class.
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image.
        # Number of boxes.
        n = x.shape[0]
        if not n:
            continue

        # Batched NMS:
        #  Classes.
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS.
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        # Limit detections.
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):

            # Merge NMS (boxes merged using weighted mean).
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4).
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def base64EncodeImage(img):
    # Takes an input image and returns a base64 encoded string representation of that image (jpg format)
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64
