_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/models/knet_citystep_s3_r50_fpn.py',
    '../_base_/datasets/cityscapes_step.py',
]

num_stages = 3
num_proposals = 100
conv_kernel_size = 1
num_proposals = 100
# load_from = "/mnt/lustre/lixiangtai/pretrained/video_knet_vis/knet_r50_city.pth"
load_from = None

work_dir = 'logger/blackhole'

runner = dict(type='EpochBasedRunner', max_epochs=8)

model = dict(
    type='KNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        type='KernelIterHead',
        merge_joint=True,
        num_thing_classes=2,
        num_stuff_classes=17,
        do_panoptic=True,
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        mask_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=19,
                num_thing_classes=2,
                num_stuff_classes=17,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'),
                    act_cfg=None
                ),
                kernel_updator_cfg=dict(
                    type='KernelUpdatorSkip',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1
                ),
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0
                ),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0
                ),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0))
            for _ in range(num_stages)
        ]
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=num_proposals,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5, stuff_max_area=4096, instance_score_thr=0.3)))
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[7, ],
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)
