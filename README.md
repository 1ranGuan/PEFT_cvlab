# Optimizing Engineering Vehicle Object Detection Performance Via Efficient Fine-Tuning  (Computer Vision Homework in HUST AIA)

The code is based on mmdetection, thanks to their efforts for the Computer Vision Society.

## prepare your environment
Follow mmdetection official install [guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

## prepare your dataset
For using mmdetection framework, we need to convert the dataset to coco format.
You can use `tools/dataset_converters/yolo2coco.py` to convert the data from yolo format to coco format.
You can also use `tools/dataset_converters/images2coco.py` to create coco test files for a batch of unlabeled images.

## Training from scratch
Maybe you should modify dataset path in `configs/_base_/datasets/cvlab_det2.py` first.

Then run
```bash
  python tools/train.py configs/cvlab/cascade-rcnn_vitb_adapt_ffn=16_expdata.py
```
for training.
