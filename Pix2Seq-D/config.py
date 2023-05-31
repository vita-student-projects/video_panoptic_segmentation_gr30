from collections import namedtuple
import torch
import torch.nn.functional as F

ImageShape = namedtuple("image_shape", ["width", "height"])

CFG = dict(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    image_size=ImageShape(width=1216, height=352),
    bits=8,
    train=dict(
        batch_size=32,
        lr=5e-4,
        accumulate_every=1,
        loss=F.mse_loss,
        freeze_backbone=False,
    ),
    pretrain=dict(
        batch_size=16,
        accumulate_every=2,
        lr=1e-4,
        freeze_backbone=False,
    ),
    misc=dict(
        frames_before=5,
        num_workers=32,
    ),
)
