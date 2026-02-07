import os, json
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from collections import defaultdict
from datasets.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_densecap import evaluate
from datasets.task_prompts import TASK_PROPMT, BOX_FORMAT


# 添加可序列化的工厂函数
def template_stats_factory():
    return {'count': 0, 'cider': 0.0}


class Dataset(ScanNetBaseDataset):
    
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        self.task_name = 'nr3d'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.split = split_set
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len
        self.eval_func = evaluate
        
        # 添加提示词选择相关属性
        self.use_prompt_selector = True  # 启用提示词选择器
        self.prompt_tracker = {}  # 用于追踪每个问题的最佳提示词
        self.current_prompt_idx = 0  # 当前使用的提示词索引
        # 使用可序列化的工厂函数
        self.prompt_performances = defaultdict(template_stats_factory)  # 提示词性能统计
        self.best_prompt_idx = 0  # 当前性能最佳的提示词索引
        
        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        ## load annotations
        assert split_set in ["train", "val"]
        
        self.scanrefer = json.load(
            open(os.path.join(BASE, 'data', 'Nr3D', f'nr3d_{split_set}.json'), 'r')
        )
        
        with open(os.path.join(BASE, 'data', 'Nr3D', f'nr3d_{split_set}.txt'), 'r') as f:
            self.scan_names = f.read().splitlines()
        
        self.annotations = self.scanrefer
        if self.split != 'train':
            self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
        self._tag_dataset(self.annotations, 'densecap')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")

    
    def _tag_dataset(self, corpus, task_name): 
        for anno in corpus:
            anno['task_name'] = task_name
        return 
    
    def _encode_box_coords(self, annotation_mask, ret_dict):
        center_normalized = ret_dict['gt_box_centers_normalized']
        size_normalized = ret_dict['gt_box_sizes_normalized']
        box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = box_normalized[annotation_mask == 1]
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        scan_name = self.annotations[idx]['scene_id']
        task_name = self.annotations[idx]['task_name']
        ret_dict = self._get_scan_data(scan_name)
        
        # 修改提示词选择逻辑
        if self.split == 'train':
            # 训练阶段有两种策略:
            # 1. 随机选择提示词（探索）
            # 2. 选择当前最佳提示词（利用）
            if random.random() < 0.7:  # 70% 概率随机选择模板（探索）
                prompt_idx = random.randrange(len(TASK_PROPMT[task_name]))
            else:  # 30% 概率使用当前最佳模板（利用）
                prompt_idx = self.best_prompt_idx
            prompt = deepcopy(TASK_PROPMT[task_name][prompt_idx])
            self.current_prompt_idx = prompt_idx  # 记录当前使用的提示词索引
        else:
            # 测试阶段使用动态选择的提示词
            # 实际提示词将由模型选择，这里只是设置一个默认值
            prompt = deepcopy(TASK_PROPMT[task_name][0])
            
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        if self.split == 'train':
            
            target_obj_id = int(self.annotations[idx]['object_id'])
            caption = ' '.join(self.annotations[idx]['token'])
            
            ## reference object
            match_mask = (ret_dict["gt_object_ids"] == target_obj_id).astype(np.float32)
            match_mask = match_mask * ret_dict["gt_box_present"]
            
            boxes = self._encode_box_coords(match_mask, ret_dict)
            response = prompt['answer'].format(locations=boxes, caption=caption)
                
            ## input_ids as labels for LLM
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            box_query = np.zeros((self.max_prompts, 8, 3))
            box_mask = np.zeros((self.max_prompts,))
            click_query = np.zeros((self.max_prompts, 3))
            click_mask = np.zeros((self.max_prompts,))
            
            if random.random() > 0.5:
                # use box to identify an object
                ref_gt_box_corner = \
                    ret_dict["gt_box_corners"][match_mask == 1].reshape(8, 3).astype(np.float32)
                box_query[0] = ref_gt_box_corner
                box_mask[0] = 1
            else:
                # use click to identify an object
                try:
                    point_clouds = ret_dict["point_clouds"][:, :3]  # x, y, z
                    object_points = point_clouds[ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
                    click_query[0] = random.choice(object_points)
                except:
                    click_query[0] = ret_dict["gt_box_centers"][match_mask == 1].reshape(3,).astype(np.float32)
                click_mask[0] = 1
            
            ret_dict['box_query'] = box_query.astype(np.float32)
            ret_dict['box_mask'] = box_mask.astype(np.float32)
            ret_dict['click_query'] = click_query.astype(np.float32)
            ret_dict['click_mask'] = click_mask.astype(np.float32)
            
            ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
            ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
            ret_dict['gradient_mask'] = \
                (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        # 添加任务特征作为辅助信息，帮助模型学习提示词选择
        if self.split == 'train':
            # 对于NR3D Dense Caption，提取对象描述特征
            caption_words = caption.lower().split()
            features = {
                'first_word': caption_words[0] if caption_words else "",  # 描述首词
                'length': len(caption_words),  # 描述长度
                'has_color': any(color in caption.lower() for color in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey']),  # 是否包含颜色词
                'has_size': any(size in caption.lower() for size in ['large', 'small', 'big', 'tiny', 'huge']),  # 是否包含尺寸词
                'has_position': any(pos in caption.lower() for pos in ['left', 'right', 'front', 'back', 'center', 'middle', 'corner']),  # 是否包含位置词
                'object_id': target_obj_id
            }
            ret_dict['caption_features'] = features
            ret_dict['caption_text'] = caption
        
        return ret_dict
        
    # 添加更新提示词性能的方法
    def update_prompt_performance(self, prompt_idx, cider_score):
        """更新特定提示词的性能统计"""
        self.prompt_performances[prompt_idx]['count'] += 1
        self.prompt_performances[prompt_idx]['cider'] += cider_score
        
        # 计算当前最佳提示词
        best_avg = -1
        for idx, stats in self.prompt_performances.items():
            if stats['count'] > 0:
                avg = stats['cider'] / stats['count']
                if avg > best_avg:
                    best_avg = avg
                    self.best_prompt_idx = idx
   