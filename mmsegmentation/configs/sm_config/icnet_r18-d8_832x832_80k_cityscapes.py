_base_ = './icnet_r50-d8_832x832_80k_cityscapes.py'
# load_from = 'pretrain/icnet_r18-d8_832x832_80k_cityscapes_20210925_225521-2e36638d.pth'
model = dict(backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18)))
