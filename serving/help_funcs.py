import cv2
import numpy as np
import torch
from torchvision.ops import (box_iou, nms)

import time
import base64

import matplotlib.path as mplPath


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

    # Timeout.
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
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

            '''
            # 기존 코드
            i, j = torch.nonzero(x[:, 5:] > conf_thres, as_tuple=False).T

            bbox = torch.Tensor(box[i])
            confidence = torch.Tensor(x[i, j + 5, None])
            class_num = j[:, None].float()

            if i.size(dim=0) == 1 and j.size(dim=0) == 1:
                bbox = bbox.reshape(1, bbox.size(dim=0))
                confidence = confidence.reshape(1, confidence.size(dim=0))

            x = torch.cat((bbox, confidence, class_num), 1)
            '''
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
        # Classes.
        c = x[:, 5:6] * (0 if agnostic else max_wh)

        # Boxes (offset by class), scores.
        boxes, scores = x[:, :4] + c, x[:, 4]

        # NMS.
        i = nms(boxes, scores, iou_thres)

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


def mns(prediction, conf_thres=0.5, iou_thres=0.6):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    xc = prediction[..., 4] > conf_thres  # candidates

    output = [None] * prediction.shape[0]
    for i, x in enumerate(prediction):  # image index, image inference
        x = x[xc[i]]  # confidence
        if not x.shape[0]:
            continue
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[..., :4])

        conf = x[..., 4].view(-1, 1)
        x = torch.cat((box, conf), 1)

        # Batched NMS
        # boxes (offset by class), scores
        boxes, scores = x[..., :4], x[..., 4]
        ind = nms(boxes, scores, iou_thres)
        output[i] = x[ind]

    return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
    img = np.float32(img)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img


def plot_result_image(json_results, img, colors, approx, big_approx, mode, inf_time=None):
    if approx != []:
        cv2.drawContours(img, [approx], 0, (0, 255, 255), 3)
        cv2.drawContours(img, [big_approx], 0, (255, 128, 0), 3)
    for bbox_list in json_results:
        for bbox in bbox_list:
            if 'deep' in mode:
                if not inf_time:
                    label = f'{bbox["track_id"]:.0f} {bbox["class_name"]}'
                elif inf_time:
                    label = f'{bbox["track_id"]:.0f} {bbox["class_name"]} Time:{inf_time:.1f}ms'
            else:
                if not inf_time:
                    label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                elif inf_time:
                    label = f'{bbox["class_name"]} {bbox["confidence"]:.2f} Time:{inf_time:.1f}ms'

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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhToxyxy(arr):
    # xytl w h to xyxy
    return [arr[0], arr[1], arr[0]+arr[2], arr[1]+arr[3]]


def base64EncodeImage(img):
    # Takes an input image and returns a base64 encoded string representation of that image (jpg format)
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

    return im_b64


# with pytorch
def results_to_json(results, classes, mode):
    # Converts yolo model output to json (list of list of dicts)
    if 'deep' in mode:
        return [
            [
                {
                    "class": int(pred[5]),
                    "class_name": classes[int(pred[5])],
                    "bbox": [int(x) for x in xywhToxyxy(pred[:4].tolist())],  # convert bbox results to int from float
                    "track_id": float(pred[4]),
                }
                for pred in result
            ]
            for result in results
        ]
    else:
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


def valid_box_idx(approx, bbox_list, is_origin, mode):
    '''
        TODO:
        현재 poly로만 convert,,,

        1. roi_box : 횡단보도의 가운데 영역을 정해야됨...
        2. return 값인 idx_list의 길이를 통해서
           traffic_control.js가 사용 할 boolen keep_green 값을 정할 수 있다. => 그릴 때 박스 거르기 가능

        return idx_list >> box_list[idx_list] 이렇게하면 다른 곳에서 필요한 박스만 쓸수 있음
    '''
    roi_box = mplPath.Path([i[0] for i in approx])

    if is_origin:
        idx_list = [idx for idx, box in enumerate(bbox_list) if (is_in_polygon(roi_box, box, mode) & (is_wheelchair(box) | is_stroller(box)))]
    else:
        idx_list = [idx for idx, box in enumerate(bbox_list) if (is_in_polygon(roi_box, box, mode) & (is_wheelchair(box) | is_stroller(box) | is_person(box)))]

    ret = True if idx_list != [] else False

    return ret, idx_list


# return index of box >> ex) [1,4,6,10]

def is_in_polygon(polygon, box, mode):
    # deepsort >> xywh
    if 'deep' in mode:
        return polygon.contains_point((box[0]+box[2]/2, box[1]+box[3]))
    # onnx >> xyxy // torch >> xyxy
    else:
        return polygon.contains_point(((box[0] + box[2]) / 2, box[3]))


def is_wheelchair(box):
    return True if box[5] == 3 else False


def is_stroller(box):
    return True if box[5] == 2 else False


def is_person(box):
    return True if box[5] == 1 else False


def approx_big(approx, size, ratio):
    M = cv2.moments(approx)

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    big_approx = [[[
        int(chk_max_min((i[0][0] - cx) * (ratio - 1) + i[0][0], size[0])),
        int(chk_max_min((i[0][1] - cy) * (ratio - 1) + i[0][1], size[1]))
    ]] for i in approx]

    return np.array(big_approx)


def chk_max_min(x, size):
    if x < 0:
        return 0
    elif x > size:
        return size
    else:
        return x


def get_inform_json(results, idx_list, keep_green, keep_red, classes):
    inform = {}

    all_box = {}
    for pred in results:
        class_num = int(pred[5])
        count = all_box.get(classes[class_num], 0)
        all_box[classes[class_num]] = count + 1

    roi_box = {}
    if idx_list != []:
        for pred in results[idx_list]:
            class_num = int(pred[5])
            count = roi_box.get(classes[class_num], 0)
            roi_box[classes[class_num]] = count + 1

    inform['log'] = {'all_box': all_box, 'roi_box': roi_box}
    inform['traffic'] = {'keep_green': keep_green, 'keep_red': keep_red}

    return inform