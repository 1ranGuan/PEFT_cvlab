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
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=3)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)