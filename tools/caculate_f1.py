import os
import numpy as np
yolo_gt_dir = '/data/yrguan/CVlab/expansion_dataset/labels'
yolo_pred_dir = '/data/yrguan/CVlab/mmdetection/result'


score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def calculate_iou(bbox_x_center, bbox_y_center, bbox_w, bbox_h, gt_bbox_x_center, gt_bbox_y_center, gt_bbox_w, gt_bbox_h):
    # 计算预测bbox的左上角坐标和右下角坐标
    bbox_x1 = bbox_x_center - bbox_w / 2
    bbox_y1 = bbox_y_center - bbox_h / 2
    bbox_x2 = bbox_x_center + bbox_w / 2
    bbox_y2 = bbox_y_center + bbox_h / 2
    # 计算gt的bbox的左上角坐标和右下角坐标
    gt_bbox_x1 = gt_bbox_x_center - gt_bbox_w / 2
    gt_bbox_y1 = gt_bbox_y_center - gt_bbox_h / 2
    gt_bbox_x2 = gt_bbox_x_center + gt_bbox_w / 2
    gt_bbox_y2 = gt_bbox_y_center + gt_bbox_h / 2
    # 计算交集的左上角坐标和右下角坐标
    inter_bbox_x1 = max(bbox_x1, gt_bbox_x1)
    inter_bbox_y1 = max(bbox_y1, gt_bbox_y1)
    inter_bbox_x2 = min(bbox_x2, gt_bbox_x2)
    inter_bbox_y2 = min(bbox_y2, gt_bbox_y2)
    # 计算交集的面积
    inter_area = max(inter_bbox_x2 - inter_bbox_x1, 0) * max(inter_bbox_y2 - inter_bbox_y1, 0)
    # 计算预测bbox和gt的bbox的面积
    bbox_area = bbox_w * bbox_h
    gt_bbox_area = gt_bbox_w * gt_bbox_h
    # 计算iou
    iou = inter_area / (bbox_area + gt_bbox_area - inter_area)
    return iou


for score_threshold in score_thresholds:
    m=0
    N=0
    n=0
    print('score_threshold: ', score_threshold)
# 遍历yolo_pred_dir下的所有文件
    for txt in os.listdir(yolo_pred_dir):
        # 读取预测的txt文件
        with open(os.path.join(yolo_pred_dir, txt), 'r') as f_pr, open(os.path.join(yolo_gt_dir, txt), 'r') as f_gt:
            lines_pr = f_pr.readlines()
            lines_gt = f_gt.readlines()
            m+=len(lines_gt)
            is_det = np.array([0 for _ in range(len(lines_gt))])
            # 遍历每一行
            for line in lines_pr:
                # 读取预测的类别、置信度、bbox
                if len(line.strip().split()) == 6:  # 有置信度
                    score_exist = True
                    category_id, bbox_x_center, bbox_y_center, bbox_w, bbox_h, score = line.strip().split()
                else:
                    score_exist = False
                    category_id, bbox_x_center, bbox_y_center, bbox_w, bbox_h = line.strip().split()
                    
                if (score_exist and float(score) > score_threshold) or score_exist == False:
                    # 如果置信度大于阈值
                    N+=1
                    # 遍历每一行
                    for gt_id,line in enumerate(lines_gt):
                        # 读取gt的类别、bbox
                        gt_category_id, gt_bbox_x_center, gt_bbox_y_center, gt_bbox_w, gt_bbox_h = line.strip().split()
                        # 如果预测的类别和gt的类别一致
                        if category_id == gt_category_id:
                            # 计算预测的bbox和gt的bbox的iou
                            iou = calculate_iou(float(bbox_x_center), float(bbox_y_center), float(bbox_w), float(bbox_h), float(gt_bbox_x_center), float(gt_bbox_y_center), float(gt_bbox_w), float(gt_bbox_h))
                            # 如果iou大于0.5
                            if iou > 0.5:
                                # 预测正确
                                is_det[gt_id] = 1
            n+=sum(is_det)

    print('m: ', m)
    print('N: ', N)
    print('n: ', n)
    print('precision: ', n/N)
    print('recall: ', n/m)
    print('f1: ', 2*n/(N+m))
