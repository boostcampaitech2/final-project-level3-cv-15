_base_ = './icnet_r50-d8_832x832_80k_cityscapes.py'
load_from = 'pretrained/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth'
model = dict(backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18)))