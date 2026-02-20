import os
import cv2
import torch
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}

# 本地模型缓存路径
_LOCAL_CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'face_detection', 'checkpoints')

def _find_local_model(name='s3fd'):
    """优先从本地加载模型，避免网络下载"""
    # 1. 直接在当前目录找
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth')
    if os.path.isfile(local):
        return local
    # 2. 在 face_detection/checkpoints 找
    if os.path.exists(_LOCAL_CHECKPOINTS):
        for f in os.listdir(_LOCAL_CHECKPOINTS):
            if f.startswith('s3fd') and f.endswith('.pth'):
                return os.path.join(_LOCAL_CHECKPOINTS, f)
    # 3. 在 TORCH_HOME/hub/checkpoints 找
    torch_home = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
    hub_cache = os.path.join(torch_home, 'hub', 'checkpoints')
    if os.path.exists(hub_cache):
        for f in os.listdir(hub_cache):
            if f.startswith('s3fd') and f.endswith('.pth'):
                return os.path.join(hub_cache, f)
    # 4. 系统默认缓存
    sys_cache = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    if os.path.exists(sys_cache):
        for f in os.listdir(sys_cache):
            if f.startswith('s3fd') and f.endswith('.pth'):
                return os.path.join(sys_cache, f)
    return None


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # 优先本地加载
        local_path = _find_local_model()
        if path_to_detector and os.path.isfile(path_to_detector):
            model_weights = torch.load(path_to_detector, map_location=device, weights_only=True)
        elif local_path:
            model_weights = torch.load(local_path, map_location=device, weights_only=True)
        else:
            model_weights = load_url(models_urls['s3fd'])

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
