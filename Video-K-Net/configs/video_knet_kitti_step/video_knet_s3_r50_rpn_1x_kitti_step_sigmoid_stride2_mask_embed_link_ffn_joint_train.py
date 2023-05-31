optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.25))))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/knet_city_step_pan_r50.pth'
resume_from = None
workflow = [('train', 1)]
num_stages = 3
num_proposals = 100
conv_kernel_size = 1
model = dict(
    type='VideoKNetQuansiEmbedFCJointTrain',
    cityscapes=False,
    kitti_step=True,
    num_thing_classes=2,
    num_stuff_classes=17,
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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='ConvKernelHead',
        num_classes=19,
        num_thing_classes=2,
        num_stuff_classes=17,
        cat_stuff_mask=True,
        conv_kernel_size=1,
        feat_downsample_stride=4,
        feat_refine_stride=1,
        feat_refine=False,
        use_binary=True,
        num_loc_convs=1,
        num_seg_convs=1,
        conv_normal_init=True,
        localization_fpn=dict(
            type='SemanticFPNWrapper',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            num_aux_convs=1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        num_proposals=100,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        num_cls_fcs=1,
        in_channels=256,
        feat_transform_cfg=None,
        loss_rank=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
        loss_seg=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=4.0)),
    roi_head=dict(
        type='VideoKernelIterHead',
        num_thing_classes=2,
        num_stuff_classes=17,
        do_panoptic=True,
        num_stages=3,
        stage_loss_weights=[1, 1, 1],
        proposal_feature_channel=256,
        mask_head=[
            dict(
                type='VideoKernelUpdateHead',
                num_classes=19,
                previous='placeholder',
                previous_type='ffn',
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
                conv_kernel_size=1,
                mask_upsample_stride=4,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)),
            dict(
                type='VideoKernelUpdateHead',
                num_classes=19,
                previous='placeholder',
                previous_type='ffn',
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
                conv_kernel_size=1,
                mask_upsample_stride=4,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)),
            dict(
                type='VideoKernelUpdateHead',
                num_classes=19,
                previous='placeholder',
                previous_type='ffn',
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
                conv_kernel_size=1,
                mask_upsample_stride=4,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0))
        ],
        with_track=True,
        merge_joint=True),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=100,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5,
                stuff_max_area=4096,
                instance_score_thr=0.25))),
    link_previous=True,
    mask_assign_stride=2,
    ignore_label=255,
    track_head=dict(
        type='QuasiDenseMaskEmbedHeadGTMask',
        num_convs=0,
        num_fcs=2,
        roi_feat_size=1,
        in_channels=256,
        fc_out_channels=256,
        embed_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
        loss_track_aux=dict(
            type='L2Loss',
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            loss_weight=1.0)),
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.35,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=5,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'),
    track_train_cfg=dict(
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
            mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
        sampler=dict(type='MaskPseudoSampler')),
    bbox_roi_extractor=None)
custom_imports = dict(
    imports=[
        'knet.det.kernel_head', 'knet.det.kernel_iter_head',
        'knet.det.kernel_update_head', 'knet.det.semantic_fpn_wrapper',
        'knet.det.dice_loss', 'knet.det.mask_hungarian_assigner',
        'knet.det.mask_pseudo_sampler', 'knet.kernel_updator',
        'knet.cross_entropy_loss', 'external.cityscapes_step', 'external.kitti_step_dvps',
        'external.dataset.dvps_pipelines.transforms',
        'external.dataset.dvps_pipelines.loading',
        'external.dataset.dvps_pipelines.tricks',
        'external.dataset.pipelines.formatting', 'knet.video.track_heads',
        'knet.video.kernel_head', 'knet.video.kernel_iter_head',
        'knet.video.kernel_update_head', 'knet.video.knet_uni_track',
        'knet.video.knet_quansi_dense',
        'knet.video.knet_quansi_dense_roi_gt_box',
        'knet.video.knet_quansi_dense_embed_fc',
        'knet.video.knet_quansi_dense_embed_fc_joint_train',
        'knet.video.knet_quansi_dense_roi_gt_box_joint_train',
        'knet.video.qdtrack.losses.l2_loss',
        'knet.video.qdtrack.losses.multipos_cross_entropy_loss',
        'knet.video.qdtrack.trackers.quasi_dense_embed_tracker'
    ],
    allow_failed_imports=False)
dataset_type = 'KITTISTEPDVPSDataset'
data_root = 'data/kitti-step'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(
        type='LoadMultiAnnotationsDirect',
        with_depth=False,
        divisor=-1,
        cherry_pick=True,
        cherry=[11, 13]),
    dict(
        type='SeqResizeWithDepth',
        img_scale=(384, 1248),
        ratio_range=[0.5, 2.0],
        keep_ratio=True),
    dict(type='SeqFlipWithDepth', flip_ratio=0.5),
    dict(
        type='SeqRandomCropWithDepth',
        crop_size=(384, 1248),
        share_params=True),
    dict(
        type='SeqNormalizeWithDepth',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=False),
    dict(type='SeqPadWithDepth', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg',
            'gt_instance_ids'
        ]),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImgDirect'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img', 'img_id', 'seq_id'],
                meta_keys=[
                    'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg', 'ori_filename'
                ])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KITTISTEPDVPSDataset',
            data_root='data/kitti-step',
            split='train',
            ref_seq_index=[-2, -1, 1, 2],
            test_mode=False,
            pipeline=[
                dict(type='LoadMultiImagesDirect'),
                dict(
                    type='LoadMultiAnnotationsDirect',
                    with_depth=False,
                    divisor=-1,
                    cherry_pick=True,
                    cherry=[11, 13]),
                dict(
                    type='SeqResizeWithDepth',
                    img_scale=(384, 1248),
                    ratio_range=[0.5, 2.0],
                    keep_ratio=True),
                dict(type='SeqFlipWithDepth', flip_ratio=0.5),
                dict(
                    type='SeqRandomCropWithDepth',
                    crop_size=(384, 1248),
                    share_params=True),
                dict(
                    type='SeqNormalizeWithDepth',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=False),
                dict(type='SeqPadWithDepth', size_divisor=32),
                dict(
                    type='VideoCollect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                        'gt_semantic_seg', 'gt_instance_ids'
                    ]),
                dict(type='ConcatVideoReferences'),
                dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
            ],
            with_depth=False)),
    val=dict(
        type='KITTISTEPDVPSDataset',
        data_root='data/kitti-step',
        split='val',
        ref_seq_index=None,
        test_mode=True,
        pipeline=[
            dict(type='LoadImgDirect'),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=[1.0],
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg', 'ori_filename', 'filename'
                        ])
                ])
        ],
        with_depth=False),
    test=dict(
        type='KITTISTEPDVPSDataset',
        data_root='data/inference_folder',
        split='val',
        ref_seq_index=None,
        test_mode=True,
        pipeline=[
            dict(type='LoadImgDirect'),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=[1.0],
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img', 'img_id', 'seq_id'],
                        meta_keys=[
                            'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg', 'ori_filename'
                        ])
                ])
        ],
        with_depth=False))
evaluation = dict()
num_thing_classes = 2
num_stuff_classes = 17
num_classes = 19
find_unused_parameters = True
work_dir = 'runs/video_knet_step-pretrained'
gpu_ids = range(0, 3)
