import sys
import os
sys.path.append(os.path.abspath('./rtdetr/src'))

import torch
import torch.nn as nn
from data.config import cfg
from layers import Detect
from layers.box_utils import encode
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy


class YolactRTDETR(nn.Module):
    """Wrapper that adapts RTDETR outputs for Yolact Detect"""

    def __init__(self):
        super().__init__()
        num_masks = cfg.mask_dim if cfg.mask_dim is not None else 32
        self.num_masks = num_masks
        self.transformer = RTDETRTransformer(num_masks=num_masks,
                                             num_classes=cfg.num_classes - 1)
        self.register_buffer('priors', torch.tensor([0.5, 0.5, 1.0, 1.0]).unsqueeze(0))
        self.detect = Detect(cfg.num_classes, bkg_label=0,
                             top_k=cfg.nms_top_k,
                             conf_thresh=cfg.nms_conf_thresh,
                             nms_thresh=cfg.nms_thresh)

    def forward(self, x):
        preds = self.transformer(x)
        pred_boxes = preds['pred_boxes']
        pred_logits = preds['pred_logits']
        mask_coefs = preds.get('mask_coefs', None)
        proto = preds.get('proto', None)

        b, q, _ = pred_boxes.shape
        priors = self.priors.expand(q, 4).to(pred_boxes.device)
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        loc = encode(boxes_xyxy, priors).view(b, q, 4)

        pred_outs = {
            'loc': loc,
            'conf': pred_logits,
            'mask': mask_coefs if mask_coefs is not None else torch.zeros(b, q, self.num_masks, device=pred_boxes.device),
            'priors': priors,
        }
        if proto is not None:
            pred_outs['proto'] = proto

        if self.training:
            preds.update(pred_outs)
            return preds
        else:
            return self.detect(pred_outs, self)

