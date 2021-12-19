norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ICNet',
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        in_channels=3,
        layer_channels=(128, 512),
        light_branch_middle_channels=32,
        psp_out_channels=512,
        out_channels=(64, 256, 256),
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False),
    neck=dict(
        type='ICNeck',
        in_channels=(64, 256, 256),
        out_channels=128,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        channels=128,
        num_convs=1,
        in_index=2,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=2,
            in_index=0,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=2,
            in_index=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CrosswalkDataset'
data_root = '/opt/ml/data/carview_crosswalk'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 540), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CrosswalkDataset',
        data_root='/opt/ml/data/carview_crosswalk',
        img_dir='images/train',
        ann_dir='gt/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(960, 540), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CrosswalkDataset',
        data_root='/opt/ml/data/carview_crosswalk',
        img_dir='images/val',
        ann_dir='gt/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CrosswalkDataset',
        data_root='/opt/ml/data/carview_crosswalk',
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='WandbLoggerHook',
            interval=100,
            init_kwargs=dict(
                project='final_project',
                entity='ptop',
                name='sm_icnet_new_dataset_w160k'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, interval=500)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')
# load_from = 'pretrained/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth'
work_dir = './work_dirs/icnet_r18-d8_832x832_80k_cityscapes'
gpu_ids = range(0, 1)
