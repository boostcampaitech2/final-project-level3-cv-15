# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True), # True 로 바꿔라
        dict(type='WandbLoggerHook', interval=100,
        init_kwargs=dict(
            project='final_project',
            entity='ptop',
            name='sm_icnet_default_train_1'
        ))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from=None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
