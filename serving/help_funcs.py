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
    model_name = 'yolov5s'
    img_size = 640
    mode = 'onnx'
    # assume input validated properly if we got here
    if model_dict[model_name] is None:
        if mode == 'torch':
            model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        elif mode == 'onnx':
            model_dict[model_name] = onnxruntime.InferenceSession(model_name+'.onnx')

    # img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR) for file in file_list]

    #img = cv2.imread("./images/bus.jpg")
    #print('임지',img.shape)
    img_batch = cv2.VideoCapture(0)
    # img_batch.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # FPS = 1000
    while True:
        # time check
        if mode == 'onnx':
            img = inference_with_onnx(img_batch, model_dict[model_name])
        elif mode == 'torch':
            img = inference_with_pt(img_batch, model_dict[model_name], img_size)
        # time check end
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' +
               img.tobytes() + b'\r\n')

        #cv2.imshow("VideoFrame", frame)

        if cv2.waitKey(100) > 0:
            break


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference_with_onnx(img_batch, model):
    # frame.shape : (Height, Width, BGR) >> to RGB img = img[:,:,::-1]
    ret, frame = img_batch.read()
    img = frame
    frame2 = frame

    frame = frame.astype(np.float32) #
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC) # 용의선상 x
    frame = np.transpose(frame, axes=(2, 0, 1)) # <<
    frame = np.expand_dims(frame, axis=0) # unsqueeze(0)


    # YCBCR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(img)
    img_ycbcr = img.convert('YCbCr')

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_ycbcr)
    img_y.unsqueeze_(0)
    frame = to_numpy(img_y)
    # === end
    img = frame2

    input_name = model.get_inputs()[0].name

    results = model.run([], {input_name: frame})

    for i in results:
        print('result >> ', i.shape)

    out = non_max_suppression(results[0], conf_thres=0.5)[0]  # BATCH SIZE is 1.

    bbox_list = onnx_results_to_json([out])[0] # >> bbox 많음

    for bbox in bbox_list:
        label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
        plot_one_box(bbox['bbox'], img, label=label, color=colors[int(bbox['class'])], line_thickness=3)

    ret, img = cv2.imencode('.jpg', img)

    return img


def inference_with_pt(img_batch, model, img_size):
    ret, frame = img_batch.read()

    results = model(frame.copy(), size=img_size)

    bbox_list = results_to_json(results, model)[0]

    for bbox in bbox_list:
        label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
        plot_one_box(bbox['bbox'], frame, label=label, color=colors[int(bbox['class'])], line_thickness=3)

    ret, img = cv2.imencode('.jpg', frame)

    return img


def onnx_results_to_json(results):
    #Converts yolo model output to json (list of list of dicts)
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


def results_to_json(results, model):
    #Converts yolo model output to json (list of list of dicts)
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


def base64EncodeImage(img):
    ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64


if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    parser.add_argument('--precache-models', action='store_true',
                        help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    if opt.precache_models:
        model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
                      for model_name in model_selection_options}

    app_str = 'server:app'  # make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)


def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.6, classes=None, agnostic=False, labels=()):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Number of classes.
    nc = prediction[0].shape[1] - 5
    print('엔시', nc)
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

        print('클래', classes)
        print('라벨', labels)
        #print('에',type((x[:, 5:] > conf_thres).nonezero()))
        # Detections matrix nx6 (xyxy, conf, cls).
        if multi_label:
            i, j = torch.nonzero(torch.Tensor(x[:, 5:] > conf_thres), as_tuple=False).T
            print('아이', type(box[i]))
            print('제이', type(x[i, j+5, None]))
            print('2', type(j[:, None].float()))
            x = torch.cat((torch.Tensor(box[i]), torch.Tensor(x[i, j + 5, None]), j[:, None].float()), 1)
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
