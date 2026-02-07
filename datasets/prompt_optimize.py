import json
import os


# 随机获取实例
from copy import deepcopy
import random

from datasets.task_prompts import TASK_PROPMT


# 针对ScanQA数据集的任务提示词模板
def get_instance_qa():
    # 读取项目根目录下ScanQA文件下ScanQA_v1.0_train.json数据集
    annotation_file = os.path.join('data', 'ScanQA', f'ScanQA_v1.0_train.json')
    # 拼接提示词内容 ### human: answer this quesiton according to the given 3D scene: "{question}" ### assistant:
    role_content = "Imagine you are observing an indoor scene. You need to accurately answer the question based on what you have observed."
    # 读取json文件
    annotations = json.load(open(annotation_file, 'r'))
    # 随机获取一个实例,进行随机索引获取，先计算实例的数量，然后随机获取一个索引
    num_scenes = len(annotations)
    # 随机获取一个索引,获取该索引对应的实例
    scene_idx = random.randint(0, num_scenes - 1)
    scene = annotations[scene_idx]
    # 随机选取任务提示词模板
    prompt = deepcopy(random.choice(TASK_PROPMT['qa']))
    prompt['instruction'] = prompt['instruction'].format(question=scene['question'])
    prompt['answer'] = prompt['answer'].format(answer=scene['answers'][0], locations='')
    # 返回提示词内容
    return role_content, prompt



