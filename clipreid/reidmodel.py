import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config.defaults import _C as clip_cfg
from .model.make_model import make_model


logger = logging.getLogger(__name__)


class ReIDModel(nn.Module):
    def __init__(
        self,
        model="clip_market1501.pt",
        model_dir="~/.cache/clip",
        num_classes=751,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.fp16 = False
        self.model = make_model(
            clip_cfg,
            num_class=num_classes,
            camera_num=1,
            view_num=1,
            clip_model_download_path=model_dir,
            device=device,
        )
        model_path = os.path.expanduser(os.path.join(model_dir, model))
        self.model.load_param(str(model_path))
        self.model.to(device).eval()
        self.device = device

    def preprocess(self, bboxes, img):
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box in bboxes:
            x1, y1, x2, y2 = box.astype("int")
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(
                crop,
                (128, 256),  # from (x, y) to (128, 256) | (w, h)
                interpolation=cv2.INTER_LINEAR,
            )

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = crop / 255

            # standardization (RGB channel order)
            crop = crop - np.array([0.485, 0.456, 0.406])
            crop = crop / np.array([0.229, 0.224, 0.225])

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        crops = torch.stack(crops, dim=0)
        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(
            dtype=torch.half if self.fp16 else torch.float, device=self.device
        )

        return crops

    def forward(self, images_batch):
        # batch to half
        if self.fp16 and images_batch.dtype != torch.float16:
            images_batch = images_batch.half()

        features = self.model(images_batch)

        if isinstance(features, (list, tuple)):
            return self.to_numpy(
                self.to_numpy(features[0])
                if len(features) == 1
                else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    @torch.no_grad()
    def get_features(self, bboxes, img):
        if bboxes.size != 0:
            crops = self.preprocess(bboxes, img)
            features = self.forward(crops)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features)
        return features
