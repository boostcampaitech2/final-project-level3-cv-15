from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random

from help_funcs import get_stream_cam

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

##############################################
#-------------GET Request Routes--------------
##############################################
@app.get("/")
def home(request: Request):
	'''
	Returns html jinja2 template render for home page form
	'''

	return templates.TemplateResponse('home.html', {
			"request": request,
			"model_selection_options": model_selection_options,
		})


@app.get("/about/")
def about_us(request: Request):
	'''
	Display about us page
	'''

	return templates.TemplateResponse('about.html', 
			{"request": request})


def video_streaming(model_name, img_size):
	return get_stream_cam(model_name, img_size)

##############################################
#------------POST Request Routes--------------
##############################################
@app.get("/video_feed")
async def detect_via_web_form():
	model_name = 'yolov5s'
	img_size = 640
	'''
	Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
	Intended for human (non-api) users.
	Returns: HTML template render showing bbox data and base64 encoded image
	'''

	return StreamingResponse(video_streaming(model_name, img_size), media_type = "multipart/x-mixed-replace; boundary=frame")


@app.post("/detect/")
async def detect_via_api(request: Request,
						file_list: List[UploadFile] = File(...), 
						model_name: str = Form(...),
						img_size: Optional[int] = Form(640),
						download_image: Optional[bool] = Form(False)):
	
	'''
	Requires an image file upload, model name (ex. yolov5s). 
	Optional image size parameter (Default 640)
	Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response
	
	Returns: JSON results of running YOLOv5 on the uploaded image. If download_image parameter is True, images with
			bboxes drawn are base64 encoded and returned inside the json response.

	Intended for API usage.
	'''

	if model_dict[model_name] is None:
		model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)
					for file in file_list]
	
	if download_image:
		#.copy() because the images are modified when running model, and we need originals when drawing bboxes later
		results = model_dict[model_name](img_batch.copy(), size = img_size) 
		json_results = results_to_json(results,model_dict[model_name])

		for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
			for bbox in bbox_list:
				label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
				plot_one_box(bbox['bbox'], img, label=label, 
						color=colors[int(bbox['class'])], line_thickness=3)

			payload = {'image_base64':base64EncodeImage(img)}
			json_results[idx].append(payload)

	else:
		#if we're not downloading the image with bboxes drawn on it, don't do img_batch.copy()
		results = model_dict[model_name](img_batch, size = img_size)
		json_results = results_to_json(results,model_dict[model_name])

	return json_results
	
##############################################
#--------------Helper Functions---------------
##############################################

def results_to_json(results, model):
	''' Converts yolo model output to json (list of list of dicts)'''
	return [
				[
					{
					"class": int(pred[5]),
					"class_name": model.model.names[int(pred[5])],
					"bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
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
	parser.add_argument('--host', default = 'localhost')
	parser.add_argument('--port', default = 8000)
	parser.add_argument('--precache-models', action='store_true', help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
	opt = parser.parse_args()

	if opt.precache_models:
		model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True) 
						for model_name in model_selection_options}
	
	app_str = 'server:app' #make the app string equal to whatever the name of this file is
	uvicorn.run(app_str, host= opt.host, port=opt.port, reload=True)
