_base_ = ['cascade-rcnn_r50_fpn_1x.py']

pretrained = '/data/yrguan/CVlab/mmdetection/configs/cvlab/sam_vit_b_01ec64.pth'  # sam pretrained

norm_cfg = dict(type='LN2d', requires_grad=True)

# fine-tuning configs 配置微调参数
tuning_config = None

model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        img_size= 1024,
        patch_size= 16,
        in_chans = 3,
        embed_dim=768,
        depth = 12,
        depths = (2,2,6,2),
        num_heads = 12,
        mlp_ratio = 4.0,
        out_chans = 256,
        qkv_bias = True,
        use_rel_pos = True,
        rel_pos_zero_init= True,
        window_size= 14,
        global_attn_indexes=[2, 5, 8, 11],
        checkpoint = pretrained,
        tuning_config = tuning_config,
        use_checkpoint=False),
        neck=dict(
            _delete_=True,
            type='SimpleFPN',
            backbone_channel=768,
            in_channels=[192, 384, 768, 768],
            out_channels=256,
            num_outs=5,
            norm_cfg=norm_cfg))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[24, 33])
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        milestones=[32, 44],
        gamma=0.1)
]
runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=3)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)