import logging

import numpy as np
import onnxruntime as rt
import cv2
import torch
import torchvision.transforms as transforms

from .model import Net

'''
def preprocess(img):
    img = cv2.resize(img, (64, 128))
    img = np.float32(img)
    img = img / 255.0
    img = img.transpose(2, 1, 0)
    img = np.expand_dims(img, axis=0)

    return img


class Extractor:
    def __init__(self, model_path) -> None:
        self.onnx_model = rt.InferenceSession(model_path)
        self.input_names = ["input_1"]
        self.output_names = ["output_1"]

    def __call__(self, im_crops):
        embs = []
        for im in im_crops:
            inp = preprocess(im)
            emb = self.onnx_model.run(self.output_names, {self.input_names[0]: inp})[0]
            embs.append(emb.squeeze())
        embs = np.array(np.stack(embs), dtype=np.float32)
        return embs
'''

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))[
            'net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)