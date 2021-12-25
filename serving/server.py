from os import write
import os
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from sse_starlette.sse import EventSourceResponse

import cv2
import numpy as np

import torch
import random

from realtime_inference import get_stream_cam
from help_funcs import (base64EncodeImage, results_to_json, plot_one_box)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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


##############################################
#------------POST Request Routes--------------
##############################################
@app.post("/video_feed")
async def detect_via_web_form(request: Request,
							file_list: List[UploadFile] = File(...)):

	'''
	Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
	Intended for human (non-api) users.
	Returns: HTML template render showing bbox data and base64 encoded image
	'''
	print(file_list)
	model_name = 'yol5_cls4_best'
	img_size = 640
	mode = 'torch_deep'
	# mode = 'onnx_deep'
	folder_name = "files"

	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	file_location = os.path.join(folder_name, file_list[0].filename)
	
	with open(file_location, "wb+") as file_object:
		file_object.write(file_list[0].file.read())

	return EventSourceResponse(get_stream_cam(model_name, img_size, mode, file_location))
	# # #assume input validated properly if we got here
	# # if model_dict[model_name] is None:
	# # 	model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

	# # img_batch = [cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)
	# # 				for file in file_list]

	# #.copy() because the images are modified when running model, and we need originals when drawing bboxes later
	# results = model_dict[model_name](img_batch.copy(), size = img_size)

	# json_results = results_to_json(results, model_dict[model_name])

	# img_str_list = []
	# #plot bboxes on the image
	# for img, bbox_list in zip(img_batch, json_results):
	# 	for bbox in bbox_list:
	# 		label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
	# 		plot_one_box(bbox['bbox'], img, label=label,
	# 				color=colors[int(bbox['class'])], line_thickness=3)

	# 	img_str_list.append(base64EncodeImage(img))

	# #escape the apostrophes in the json string representation
	# encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

	# return templates.TemplateResponse('show_results.html', {
	# 		'request': request,
	# 		'bbox_image_data_zipped': zip(img_str_list,json_results), #unzipped in jinja2 template
	# 		'bbox_data_str': encoded_json_results,
	# 	})

@app.get("/video_feed")
async def detect_via_web_form():
	# 하드 코딩 된 부분 >> 인자 받아보게 변경, yaml or from html
	print("여긴듯?")
	model_name = 'yol5_cls4_best'
	# model_name = 'sm_yolov5n_t1'
	img_size = 640
	mode = 'torch_deep'

	'''
	Requires an image file upload, model name (ex. yolov5s). Optional image size parameter (Default 640).
	Intended for human (non-api) users.
	Returns: HTML template render showing bbox data and base64 encoded image
	'''
	#return StreamingResponse(get_stream_cam(model_name, img_size, mode), media_type="multipart/x-mixed-replace; boundary=frame")
	return EventSourceResponse(get_stream_cam(model_name, img_size, mode))


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
