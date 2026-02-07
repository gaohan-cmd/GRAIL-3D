import os, time, json
import torch
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    is_primary, 
    barrier,
    all_gather_dict,
    is_distributed
)

def template_stats_factory():
    return {'count': 0, 'cider': 0.0}

def score_captions(corpus: dict, candidates: dict):
    
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    
    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }
    
    message = '\n'.join([
        "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
        ),
        "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
        ),
        "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
        ),
        "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
        ),
        "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            cider[0], max(cider[1]), min(cider[1])
        ),
        "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            rouge[0], max(rouge[1]), min(rouge[1])
        ),
        "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
            meteor[0], max(meteor[1]), min(meteor[1])
        )
    ])
    
    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric


def prepare_corpus(raw_data, max_len: int=30) -> dict:
    # helper function to prepare ground truth captions
    corpus = defaultdict(list)
    object_id_to_name = defaultdict(lambda:'unknown')
    
    for data in raw_data:
        
        (         scene_id,         object_id,         object_name
        ) = data["scene_id"], data["object_id"], data["object_name"]
        
        # parse language tokens
        token = data["token"][:max_len]
        description = " ".join(["sos"] + token + ["eos"])
        key = f"{scene_id}|{object_id}|{object_name}"
        object_id_to_name[f"{scene_id}|{object_id}"] = object_name
        
        corpus[key].append(description)
        
    return corpus, object_id_to_name



@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    
    # 初始化评估模式标志
    use_adaptive_prompt = True  # 默认启用自适应提示词
    prompt_tracker = None
    
    # 检查数据集是否支持提示词选择
    if hasattr(dataset_loader.dataset, 'use_prompt_selector') and dataset_loader.dataset.use_prompt_selector:
        use_adaptive_prompt = True
        prompt_tracker = dataset_loader.dataset.prompt_tracker
    
    # prepare ground truth caption labels
    print("preparing corpus...")
    scene_list = dataset_loader.dataset.scan_names
    corpus, object_id_to_name = prepare_corpus(dataset_loader.dataset.scanrefer)
    task_name = dataset_loader.dataset.task_name
    
    # 用于记录样本ID、描述和模型内部选择的提示词等信息
    sample_ids = []
    captions = []
    caption_features = []
    selected_templates = []  # 记录模型选择的提示词模板索引
    caption_keys = []  # 记录描述的唯一键，用于后续性能更新
    
    ### initialize and prepare for evaluation
    tokenizer = dataset_loader.dataset.tokenizer
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    
    model.eval()
    barrier()
    
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    # 确定评估模式
    expert_mode = getattr(args, 'expert_prompt_mode', 'model_select')  # 'model_select', 'tracker_select', 'ensemble'
    if is_primary():
        logout(f"Using '{expert_mode}' mode for template selection with adaptive_prompt={use_adaptive_prompt}")
    
    candidates = {'caption': OrderedDict({}), 'iou': defaultdict(float)}
    
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        
        curr_time = time.time()
        # 数据移动到设备时处理不同类型的数据
        for key in batch_data_label:
            if isinstance(batch_data_label[key], torch.Tensor):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            elif isinstance(batch_data_label[key], list) or isinstance(batch_data_label[key], dict):
                # 列表或字典类型数据不需要移动到设备
                pass
            else:
                # 对于其他类型的数据，可能需要特殊处理
                pass
        
        # 设置批次的自适应提示词标志
        if use_adaptive_prompt:
            batch_data_label['use_adaptive_prompt'] = torch.ones(batch_data_label['instruction'].shape[0], dtype=torch.int64).to(net_device)
        
        model_input = {
            'point_clouds': batch_data_label['point_clouds'],
            'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'],
            'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'],
            'instruction': batch_data_label['instruction'],
            'instruction_mask': batch_data_label['instruction_mask'],
            'qformer_input_ids': batch_data_label['qformer_input_ids'],
            'qformer_attention_mask': batch_data_label['qformer_attention_mask'],
            'use_adaptive_prompt': batch_data_label.get('use_adaptive_prompt', None)
        }
        outputs = model(model_input, is_eval=True, task_name='dense-cap')
        
        # 收集模型选择的提示词信息
        if 'selected_templates' in outputs:
            for template_idx in outputs['selected_templates']:
                selected_templates.append(int(template_idx))
        
        outputs = dict(
            box_corners=outputs["box_corners"],
            sem_cls_prob=outputs["sem_cls_prob"],
            objectness_prob=outputs["objectness_prob"],
            output_ids=outputs["output_ids"],
            sem_cls_logits=outputs["sem_cls_logits"],
        )
        
        # 收集结果（分布式情况下）
        if is_distributed() and not use_adaptive_prompt:
            outputs = all_gather_dict(outputs)
            batch_data_label = all_gather_dict(batch_data_label)
        
        ### match objects
        batch_size, MAX_NUM_OBJ, _, _ = batch_data_label["gt_box_corners"].shape
        _, nqueries, _, _ = outputs["box_corners"].shape
        
        match_box_ious = box3d_iou_batch_tensor(    # batch, nqueries, MAX_NUM_OBJ
            (outputs["box_corners"]
             .unsqueeze(2)
             .repeat(1, 1, MAX_NUM_OBJ, 1, 1)
             .view(-1, 8, 3)
             ),
            (batch_data_label["gt_box_corners"]
             .unsqueeze(1)
             .repeat(1, nqueries, 1, 1, 1)
             .view(-1, 8, 3)
             )
        ).view(batch_size, nqueries, MAX_NUM_OBJ)
        match_box_ious, match_box_idxs = match_box_ious.max(-1) # batch, nqueries
        match_box_idxs = torch.gather(
            batch_data_label['gt_object_ids'], 1, 
            match_box_idxs
        ) # batch, nqueries
        
        # ---- Checkout bounding box ious and semantic logits
        good_bbox_masks = match_box_ious > args.test_min_iou     # batch, nqueries
        good_bbox_masks &= outputs["sem_cls_logits"].argmax(-1) != (
            outputs["sem_cls_logits"].shape[-1] - 1
        )
        
        # ---- add nms to get accurate predictions
        nms_bbox_masks = parse_predictions( # batch x nqueries
            outputs["box_corners"], 
            outputs['sem_cls_prob'], 
            outputs['objectness_prob'], 
            batch_data_label['point_clouds']
        )
        nms_bbox_masks = torch.from_numpy(nms_bbox_masks).long() == 1
        good_bbox_masks &= nms_bbox_masks.to(good_bbox_masks.device)
        
        good_bbox_masks = good_bbox_masks.cpu().tolist()
        
        output_ids = outputs["output_ids"]  # batch x nqueries x max_length
        batch_captions = tokenizer.batch_decode(
            output_ids.reshape(-1, output_ids.shape[-1]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        batch_captions = [
            [
                ('sos ' + batch_captions[batch_id * nqueries + prop_id] + ' eos').replace('  ', ' ') \
                    for prop_id in range(nqueries)
            ] \
            for batch_id in range(batch_size)
        ]
            
        match_box_idxs = match_box_idxs.cpu().tolist()
        match_box_ious = match_box_ious.cpu().tolist()
        
        # 获取当前批次中样本的ID
        sample_index = batch_data_label['scan_idx'].cpu().tolist()
        
        ### calculate measurable indicators on captions
        for idx, scene_id in enumerate(sample_index):
            scene_name = scene_list[scene_id]
            sample_ids.append(scene_id)
            
            for prop_id in range(nqueries):

                if good_bbox_masks[idx][prop_id] is False:
                    continue
                
                match_obj_id = match_box_idxs[idx][prop_id]
                match_obj_iou = match_box_ious[idx][prop_id]
                
                object_name = object_id_to_name[f"{scene_name}|{match_obj_id}"]
                key = f"{scene_name}|{match_obj_id}|{object_name}"
                
                if match_obj_iou > candidates['iou'][key]:
                    candidates['iou'][key] = match_obj_iou
                    candidates['caption'][key] = [
                        batch_captions[idx][prop_id]
                    ]
                    
                    # 记录用于性能更新的信息
                    caption_keys.append(key)
                    captions.append(batch_captions[idx][prop_id])
                    
                    # 提取描述特征
                    caption_text = batch_captions[idx][prop_id].replace('sos ', '').replace(' eos', '')
                    caption_words = caption_text.lower().split()
                    features = {
                        'first_word': caption_words[0] if caption_words else "",
                        'length': len(caption_words),
                        'has_color': any(color in caption_text.lower() for color in ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey']),
                        'has_size': any(size in caption_text.lower() for size in ['large', 'small', 'big', 'tiny', 'huge']),
                        'has_position': any(pos in caption_text.lower() for pos in ['left', 'right', 'front', 'back', 'center', 'middle', 'corner']),
                        'object_id': match_obj_id
                    }
                    caption_features.append(features)
                    # DEBUG: checkout how many matched bounding boxes
                    # candidates[key] = ["this is a valid match!"]
                    
        # Memory intensive as it gathers point cloud GT tensor across all ranks
        time_delta.update(time.time() - curr_time)
        
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            adaptive_info = "(Dynamic Expert)" if use_adaptive_prompt else ""
            logout(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; "
                f"{adaptive_info} "
                f"Evaluating on iter: {curr_train_iter}; "
                f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )
        barrier()
    # end of forward pass traversion
    
    ### message out
    missing_proposals = len(corpus.keys() - candidates['caption'].keys())
    total_captions = len(corpus.keys())
    
    
    ### make up placeholders for undetected bounding boxes
    for missing_key in (corpus.keys() - candidates['caption'].keys()):
        candidates['caption'][missing_key] = ["sos eos"]
    
    # find annotated objects in scanrefer
    candidates_final = OrderedDict([
        (key, value) for key, value in sorted(candidates['caption'].items()) \
            if not key.endswith("unknown")
    ])
    score_per_caption, message, eval_metric = score_captions(
        OrderedDict([(key, corpus[key]) for key in candidates_final]), candidates_final
    )
    
    # 增加样本ID、描述列表和特征到返回指标中，以便训练过程更新提示词性能
    eval_metric['sample_ids'] = sample_ids
    eval_metric['captions'] = captions
    eval_metric['caption_features'] = caption_features
    eval_metric['cider_scores'] = score_per_caption['cider']
    eval_metric['caption_keys'] = caption_keys
    
    # 若有模型选择的提示词信息，也添加到评估指标中
    logout(f"selected_templates: {selected_templates}")
    if selected_templates:
        # 将selected_templates写入到文件
        with open(os.path.join(args.checkpoint_dir, "densecap_selected_templates.json"), "w") as f:
            json.dump(selected_templates, f, indent=4)
        eval_metric['selected_templates'] = selected_templates
        
        # 按描述类型分析提示词选择情况
        template_by_captype = defaultdict(list)
        for i, caption in enumerate(captions):
            if i < len(selected_templates):
                # 根据描述特征分类
                cap_type = "unknown"
                if i < len(caption_features):
                    features = caption_features[i]
                    if features['has_color']:
                        cap_type = "color_based"
                    elif features['has_position']:
                        cap_type = "position_based"
                    elif features['has_size']:
                        cap_type = "size_based"
                    else:
                        cap_type = "general"
                template_by_captype[cap_type].append(selected_templates[i])
        
        # 计算每种描述类型最常用的提示词
        template_captype_stats = {}
        for cap_type, templates in template_by_captype.items():
            # 计数
            counter = defaultdict(int)
            for t in templates:
                counter[t] += 1
            # 找出最常用的
            most_common = max(counter.items(), key=lambda x: x[1]) if counter else (0, 0)
            template_captype_stats[cap_type] = {
                'most_common': most_common[0],
                'count': most_common[1],
                'total': len(templates),
                'distribution': dict(counter)
            }
        
        eval_metric['template_by_captype'] = template_captype_stats
        
        # 如果在训练阶段，更新提示词性能
        logout(f"curr_epoch: {curr_epoch},如果在训练阶段，更新提示词性能")
        logout(f"hasattr(model, 'update_template_performance'): {hasattr(model, 'update_template_performance')}")
        logout(f"hasattr(dataset_loader.dataset, 'update_prompt_performance'): {hasattr(dataset_loader.dataset, 'update_prompt_performance')}")
        if curr_epoch >= 0 and hasattr(model, 'update_template_performance') and hasattr(dataset_loader.dataset, 'update_prompt_performance'):
            # 更新模型中的提示词性能
            if is_primary():
                logout("\nUpdating template performance statistics...")
                
            for i, (key, template_idx) in enumerate(zip(caption_keys, selected_templates)):
                if i < len(score_per_caption['cider']):
                    cider_score = score_per_caption['cider'][i]
                    # 更新模型中的统计
                    if hasattr(model, 'module'):  # 分布式情况
                        model.module.update_template_performance(template_idx, cider_score)
                    else:
                        model.update_template_performance(template_idx, cider_score)
                    
                    # 更新数据集中的统计
                    dataset_loader.dataset.update_prompt_performance(template_idx, cider_score)
    
    if is_primary():
        logout(
            f"\n----------------------Evaluation-----------------------\n"
            f"INFO: iou@{args.test_min_iou} matched proposals: "
            f"[{total_captions - missing_proposals} / {total_captions}], "
        )
        logout(f"Using {'Dynamic Expert Prompt Selection' if use_adaptive_prompt else 'Static Prompt'}")
        logout(message)
        
        # 输出提示词选择统计
        logout(f"selected_templates: {selected_templates}")
        if selected_templates:
            template_counter = defaultdict(int)
            for t in selected_templates:
                template_counter[t] += 1
                
            total = len(selected_templates)
            logout("\nTemplate Selection Statistics:")
            for template_idx, count in sorted(template_counter.items()):
                logout(f"Template {template_idx}: {count}/{total} ({count/total*100:.1f}%)")
            
            # 输出描述类型与提示词的关系
            if 'template_by_captype' in eval_metric:
                logout("\nTemplate Selection by Caption Type:")
                for cap_type, stats in sorted(eval_metric['template_by_captype'].items(), 
                                           key=lambda x: x[1]['total'], reverse=True):
                    most_common = stats['most_common']
                    percent = stats['count'] / stats['total'] * 100 if stats['total'] > 0 else 0
                    logout(f"Type '{cap_type}': Template {most_common} ({percent:.1f}%)")
                    
            # 输出模型和数据集中的提示词性能统计
            if hasattr(model, 'template_performances'):
                model_perf = model.template_performances if not hasattr(model, 'module') else model.module.template_performances
                logout("\nModel Template Performance:")
                for idx, stats in sorted(model_perf.items()):
                    avg_score = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                    logout(f"Template {idx}: Avg CIDEr = {avg_score:.4f} (used {stats['count']} times)")
                best_idx = model.best_template_idx if not hasattr(model, 'module') else model.module.best_template_idx
                logout(f"Model's Best Template: {best_idx}")
                
            if hasattr(dataset_loader.dataset, 'prompt_performances'):
                logout("\nDataset Template Performance:")
                for idx, stats in sorted(dataset_loader.dataset.prompt_performances.items()):
                    avg_score = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                    logout(f"Template {idx}: Avg CIDEr = {avg_score:.4f} (used {stats['count']} times)")
                logout(f"Dataset's Best Template: {dataset_loader.dataset.best_prompt_idx}")
        
        with open(os.path.join(args.checkpoint_dir, task_name + "_densecap_corpus_val.json"), "w") as f: 
            json.dump(corpus, f, indent=4)
        
        with open(os.path.join(args.checkpoint_dir, task_name + "_densecap_pred_val.json"), "w") as f:
            json.dump(candidates_final, f, indent=4)
        
        with open(os.path.join(args.checkpoint_dir, task_name + "_densecap_pred_gt_val.json"), "w") as f:
            pred_gt_val = {}
            for scene_object_id, scene_object_id_key in enumerate(candidates_final):
                entry = {
                    'pred': candidates_final[scene_object_id_key],
                    'gt': corpus[scene_object_id_key],
                    'score': {
                        'bleu-1': score_per_caption['bleu-1'][scene_object_id],
                        'bleu-2': score_per_caption['bleu-2'][scene_object_id],
                        'bleu-3': score_per_caption['bleu-3'][scene_object_id],
                        'bleu-4': score_per_caption['bleu-4'][scene_object_id],
                        'CiDEr': score_per_caption['cider'][scene_object_id],
                        'rouge': score_per_caption['rouge'][scene_object_id],
                        'meteor': score_per_caption['meteor'][scene_object_id]
                    }
                }
                
                # 如果有模型选择的提示词，也添加到详细信息中
                if scene_object_id < len(selected_templates):
                    entry['selected_template'] = selected_templates[scene_object_id]
                    
                pred_gt_val[scene_object_id_key] = entry
                
            json.dump(pred_gt_val, f, indent=4)
            
        # 如果使用了提示词选择器，保存提示词选择统计
        if use_adaptive_prompt and selected_templates:
            template_stats = {
                'counts': dict(template_counter),
                'by_caption_type': eval_metric.get('template_by_captype', {}),
                'epoch': curr_epoch,
                'train_iter': curr_train_iter
            }
            
            # 添加性能数据
            if hasattr(model, 'template_performances'):
                model_perf = model.template_performances if not hasattr(model, 'module') else model.module.template_performances
                performance = {}
                for idx, stats in model_perf.items():
                    avg_cider = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                    performance[str(idx)] = {
                        'count': stats['count'],
                        'cider': stats['cider'],
                        'avg_cider': avg_cider
                    }
                template_stats['performance'] = performance
            
            template_stats_file = os.path.join(
                args.checkpoint_dir, 
                f"densecap_template_selection_stats_epoch{curr_epoch}_iter{curr_train_iter}.json"
            )
            with open(template_stats_file, "w") as f:
                json.dump(template_stats, f, indent=4)
    
    eval_metrics = {
        metric + f'@{args.test_min_iou}': score \
            for metric, score in eval_metric.items() if isinstance(score, (int, float))
    }
    # 保留非数值型的评估指标
    for key, value in eval_metric.items():
        if not isinstance(value, (int, float)):
            eval_metrics[key] = value
            
    return eval_metrics