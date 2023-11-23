
import os
import json

## 计算coco格式数据集每一类的数量

# 1. 读取coco格式的json文件
json_file = '/data/yrguan/CVlab/dataset/annotations/val.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# 2. 统计每一类的数量
categories = data['categories']
images = data['images']
annotations = data['annotations']

# 2.1 统计每一类的数量
category_num = {}
for category in categories:
    category_num[category['name']] = 0
for annotation in annotations:
    category_num[categories[annotation['category_id']]['name']] += 1   

print(category_num)


