_base_ = [
    '../_base_/models/cascade-rcnn_r50_fpn.py',
    '../_base_/datasets/cvlab_det.py','../_base_/default_runtime.py'
]
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=700),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=36,
#         by_epoch=True,
#         milestones=[27, 33],
#         gamma=0.1)
# ]
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)