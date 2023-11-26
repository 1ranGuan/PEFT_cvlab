#本代码把生成的coco格式的json文件转换成yolo格式的txt文件
import json

NEED_SCORE = True

if NEED_SCORE==False:  # 不用置信度 设置阈值
    score_threshold = 0.8

coco_annotation_file = '/data/yrguan/CVlab/expansion_dataset/annotations/test.json'
coco_result_file = 'work_dirs/coco_detection/result.bbox.json'
out_dir = 'result/'
#读取coco格式的json文件
with open(coco_annotation_file, 'r') as f:
    coco_annotation = json.load(f)
    img_msgs = coco_annotation['images']
    # 构建img_id和img_name的映射关系
    img_id2info = {}
    for img_msg in img_msgs:
        img_info = dict(file_name=img_msg['file_name'], width=img_msg['width'], height=img_msg['height'])
        img_id2info[img_msg['id']]= img_info

#读取coco输出的json文件
with open(coco_result_file, 'r') as f:
    coco_result = json.load(f)
    for result in coco_result:
        img_id = result['image_id']
        print(img_id)
        img_info = img_id2info[img_id]
        img_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        bbox = result['bbox']
        category_id = result['category_id']
        score = result['score']
        bbox_x = bbox[0]
        bbox_y = bbox[1]
        bbox_w = bbox[2]
        bbox_h = bbox[3]
        bbox_x_center = bbox_x + bbox_w / 2
        bbox_y_center = bbox_y + bbox_h / 2
        bbox_w = bbox_w
        bbox_h = bbox_h
        #把bbox转换成yolo格式的txt文件
        with open(out_dir + img_name.replace('.jpg', '.txt'), 'a') as f:
            if NEED_SCORE:
                f.write(f'{category_id} {bbox_x_center/img_width} {bbox_y_center/img_height} {bbox_w/img_width} {bbox_h/img_height} {score}\n')
            else:
                if score > score_threshold:
                    f.write(f'{category_id} {bbox_x_center/img_width} {bbox_y_center/img_height} {bbox_w/img_width} {bbox_h/img_height}\n')
