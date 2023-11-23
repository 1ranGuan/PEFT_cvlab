_base_ = ['cascade-rcnn_r50_fpn_1x.py']

pretrained = '/data/yrguan/CVlab/sam_vit_h_4b8939.pth'  # sam pretrained

norm_cfg = dict(type='LN2d', requires_grad=True)

# fine-tuning configs 配置微调参数
tuning_config = dict(
    # AdaptFormer
    SEadapter=True,
    adaptmix=False,
    ffn_adapt=False,
    ffn_option="parallel",
    ffn_adapter_layernorm_option="none",
    ffn_adapter_init_option="lora",
    ffn_adapter_scalar="0.1",
    ffn_num=40,
    d_model=1280,
    # VPT related
    vpt_on=False,
    vpt_num=1
)
        # encoder_embed_dim=1280,
        # encoder_depth=32,
        # encoder_num_heads=16,
        # encoder_global_attn_indexes=[7, 15, 23, 31],
model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        img_size= 1024,
        patch_size= 16,
        in_chans = 3,
        embed_dim=1280,
        depth = 32,
        depths = (2,2,6,2),
        num_heads = 16,
        mlp_ratio = 4.0,
        out_chans = 256,
        qkv_bias = True,
        use_rel_pos = True,
        rel_pos_zero_init= True,
        window_size= 14,
        global_attn_indexes=[7, 15, 23, 31],
        checkpoint = pretrained,
        tuning_config = tuning_config,
        use_checkpoint=True),
        neck=dict(
            _delete_=True,
            type='SimpleFPN',
            backbone_channel=1280,
            in_channels=[320, 640, 1280, 1280],
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
        weight_decay=0.1
    ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=3)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)