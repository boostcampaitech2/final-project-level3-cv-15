from fastapi import FastAPI, Form, File, UploadFile
from typing import List, Optional

import cv2

import torch

import random

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

def get_stream_video(model_name: str = Form(...),
                        img_size: int = Form(640)):
    '''
    Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
    Intended for human (non-api) users.
    Returns: HTML template render showing bbox data and base64 encoded image
    '''

    # assume input validated properly if we got here
    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR) for file in file_list]

    img_batch = cv2.VideoCapture(0)
    # img_batch.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # FPS = 1000
    while True:
        ret, frame = img_batch.read()

        results = model_dict[model_name](frame.copy(), size=img_size)
        bbox_list = results_to_json(results, model_dict[model_name])[0]

        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], frame, label=label, color=colors[int(bbox['class'])], line_thickness=3)

        ret, img = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' +
               img.tobytes() + b'\r\n')

        #cv2.imshow("VideoFrame", frame)

        if cv2.waitKey(1) > 0:
            break


def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
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
