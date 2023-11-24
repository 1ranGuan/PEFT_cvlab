_base_ = ['cascade-rcnn_vitb_adapt_ffn=16.py']



model=dict(
    backbone=dict(
        tuning_config=dict(
            ffn_num=0
        )
    )
)