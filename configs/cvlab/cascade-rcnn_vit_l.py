_base_ = ['cascade-rcnn_r50_fpn_1x.py']

pretrained = '/data/yrguan/CVlab/sam_vit_l_0b3195.pth'  # sam pretrained

norm_cfg = dict(type='LN2d', requires_grad=True)

# fine-tuning configs 配置微调参数
tuning_config = dict(
    # AdaptFormer
    SEadapter=False,
    adaptmix=False,
    ffn_adapt=True,
    ffn_option="parallel",
    ffn_adapter_layernorm_option="none",
    ffn_adapter_init_option="lora",
    ffn_adapter_scalar="0.1",
    ffn_num=64,
    d_model=1024,
    # VPT related
    vpt_on=False,
    vpt_num=1
)
# encoder_embed_dim=1024,
# encoder_depth=24,
# encoder_num_heads=16,
# encoder_global_attn_indexes=[5, 11, 17, 23],
model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        img_size= 1024,
        patch_size= 16,
        in_chans = 3,
        embed_dim=1024,
        depth = 24,
        depths = (2,2,6,2),
        num_heads = 16,
        mlp_ratio = 4.0,
        out_chans = 256,
        qkv_bias = True,
        use_rel_pos = True,
        rel_pos_zero_init= True,
        window_size= 14,
        global_attn_indexes=[5, 11, 17, 23],
        checkpoint = pretrained,
        tuning_config = tuning_config,
        use_checkpoint=True),
        neck=dict(
            _delete_=True,
            type='SimpleFPN',
            backbone_channel=1024,
            in_channels=[256, 512, 1024, 1024],
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

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=3)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)