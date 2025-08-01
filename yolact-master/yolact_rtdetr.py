import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external', 'rtdetr_pytorch', 'src')))

from zoo.rtdetr import HybridEncoder, RTDETRTransformer
from backbone import construct_backbone
from utils.functions import make_net
from data.config import cfg, mask_type


class YolactRTDETR(nn.Module):
    """YOLACT model using RT-DETR for detection.

    This wraps an :class:`RTDETR` detector and adds the prototype mask branch
    from the original YOLACT architecture. The RT-DETR decoder is modified to
    output mask coefficients, which are then combined with the prototypes to
    produce instance masks during post-processing.
    """

    def __init__(self):
        super().__init__()

        # Backbone for RT-DETR
        self.backbone = construct_backbone(cfg.backbone)

        # RT-DETR encoder and decoder. The decoder outputs mask coefficients
        # (cfg.mask_dim) for each query.
        self.encoder = HybridEncoder(in_channels=self.backbone.channels[-3:],
                                     feat_strides=[8, 16, 32])
        self.decoder = RTDETRTransformer(num_masks=cfg.mask_dim,
                                         num_classes=cfg.num_classes,
                                         feat_channels=[256, 256, 256],
                                         feat_strides=[8, 16, 32])

        # Prototype network used to generate mask bases (same as YOLACT)
        in_channels = 3 if cfg.mask_proto_src is None else self.backbone.channels[cfg.mask_proto_src]
        self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

    def forward(self, x, targets=None):
        """Perform a forward pass with mask coefficient output."""

        # Run backbone once
        feats = self.backbone(x)

        # Prototype branch
        proto_x = x if cfg.mask_proto_src is None else feats[cfg.mask_proto_src]
        proto_out = self.proto_net(proto_x)
        proto_out = cfg.mask_proto_prototype_activation(proto_out)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        # Detection branch using RT-DETR
        enc_out = self.encoder(feats)
        det_out = self.decoder(enc_out, targets)
        det_out['proto'] = proto_out
        return det_out
