import os, sys, time, math, json, importlib
import random

import numpy as np
import torch
import datetime
from collections import defaultdict, OrderedDict

import utils.capeval.bleu.bleu as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.box_util import box3d_iou_batch_tensor
from utils.ap_calculator import APCalculator
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.proposal_parser import parse_predictions
from utils.dist import (
    init_distributed,
    is_distributed,
    is_primary,
    get_rank,
    barrier,
    all_reduce_average,
    all_gather_dict
)


def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, args):
        exp_name = os.path.split(args.checkpoint_dir)[-1]
        self.logger = open(os.path.join(args.checkpoint_dir, f'{exp_name}-logger.log'), 'a')

    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
            curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
            and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
                (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def do_train(
        args,
        model,
        model_no_ddp,
        optimizer,
        dataset_config,
        dataloaders,
        best_val_metrics=dict()
):
    logout = Logger(args)

    if is_primary():
        logout(f"call with args: {args}")
        logout(f"{model}")

    curr_iter = args.start_epoch * len(dataloaders['train'])
    max_iters = args.max_epoch * len(dataloaders['train'])
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    max_tolerant_nan = 4
    curr_nan_times = 0

    for curr_epoch in range(args.start_epoch, args.max_epoch):

        if is_distributed():
            dataloaders["train_sampler"].set_epoch(curr_epoch)

        for batch_idx, batch_data_label in enumerate(dataloaders['train']):

            curr_time = time.time()

            curr_iter = curr_epoch * len(dataloaders['train']) + batch_idx
            curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
            for key in batch_data_label:
                if isinstance(batch_data_label[key], torch.Tensor):
                    batch_data_label[key] = batch_data_label[key].to(net_device)

            # 对于使用提示词选择器的训练，添加标志
            # 检查训练数据集是否包含支持提示词选择的任务
            has_prompt_selector = False
            for dataset in dataloaders['train'].dataset.datasets if hasattr(dataloaders['train'].dataset, 'datasets') else [dataloaders['train'].dataset]:
                if hasattr(dataset, 'use_prompt_selector') and dataset.use_prompt_selector:
                    has_prompt_selector = True
                    break

            if has_prompt_selector:
                batch_data_label['use_adaptive_prompt'] = torch.ones(batch_data_label['instruction'].shape[0], dtype=torch.int64).to(net_device)

            # Forward pass
            optimizer.zero_grad()

            outputs = model(batch_data_label, is_eval=False)
            loss = outputs['loss']
            loss = all_reduce_average(loss)

            if not math.isfinite(loss.item()):
                if curr_nan_times < max_tolerant_nan:
                    logout("Loss in not finite. Skip this training step.")
                    curr_nan_times += 1
                    continue
                else:
                    logout("Loss in not finite. Terminate training.")
                    exit(-1)
            curr_nan_times = 0

            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()

            time_delta.update(time.time() - curr_time)
            loss_avg.update(loss.item())

            # logging
            if is_primary() and curr_iter % args.log_every == 0:
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # 添加提示词选择器状态信息
                prompt_info = ""
                if hasattr(model, 'best_template_idx') and not hasattr(model, 'module'):
                    prompt_info = f"Best Template: {model.best_template_idx}; "
                elif hasattr(model, 'module') and hasattr(model.module, 'best_template_idx'):
                    prompt_info = f"Best Template: {model.module.best_template_idx}; "
                
                logout(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; "
                    f"Iter [{curr_iter}/{max_iters}]; "
                    f"Loss {loss_avg.avg:0.2f}; "
                    f"{prompt_info}"
                    f"LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; "
                    f"ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )

            barrier()
            # save ckpt
            if is_primary() and (curr_iter + 1) % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    curr_epoch,
                    args,
                    best_val_metrics,
                    filename=f"checkpoint_{(curr_iter + 1) // 1000}k.pth",
                )

            # eval
            if (curr_iter + 1) % args.eval_every_iteration == 0 \
                    and (curr_iter + 1) > args.start_eval_after:

                eval_metrics = {}
                model.eval()
                for test_loader in dataloaders['test']:
                    task_metrics = test_loader.dataset.eval_func(
                        args,
                        curr_epoch,
                        model,
                        dataset_config,
                        test_loader,
                        logout,
                        curr_train_iter=curr_iter
                    )
                    eval_metrics.update(task_metrics)
                    
                    # 修改评估后的提示词性能更新部分
                    # 如果评估返回了提示词选择信息，更新训练数据集的提示词性能
                    if 'selected_templates' in task_metrics and 'cider_scores' in task_metrics:
                        if is_primary():
                            logout("Updating training dataset prompt performance based on evaluation results...")
                        
                        # 从ConcatDataset中找到对应的数据集
                        for dataset in dataloaders['train'].dataset.datasets if hasattr(dataloaders['train'].dataset, 'datasets') else [dataloaders['train'].dataset]:
                            # 支持QA、Dense Caption(ScanRefer)和NR3D任务
                            if (hasattr(dataset, 'task_name') and dataset.task_name == 'scanqa' and 'question_keys' in task_metrics) or \
                               (hasattr(dataset, 'task_name') and dataset.task_name in ['scanrefer', 'nr3d'] and 'caption_keys' in task_metrics):
                                # 遍历评估结果，更新提示词性能
                                keys_field = 'question_keys' if dataset.task_name == 'scanqa' else 'caption_keys'
                                if keys_field in task_metrics:
                                    for i, (template_idx, cider_score) in enumerate(zip(task_metrics['selected_templates'], task_metrics['cider_scores'])):
                                        if i < len(task_metrics[keys_field]):
                                            dataset.update_prompt_performance(int(template_idx), float(cider_score))
                                    
                                    if is_primary():
                                        logout(f"Updated training dataset prompt performance for {dataset.task_name}. Best prompt index is now: {dataset.best_prompt_idx}")
                                    break
                
                model.train()

                if not best_val_metrics or (
                        best_val_metrics[args.criterion] < eval_metrics[args.criterion]
                ):
                    best_val_metrics = eval_metrics
                    filename = "checkpoint_best.pth"
                    save_checkpoint(
                        args.checkpoint_dir,
                        model_no_ddp,
                        optimizer,
                        curr_epoch,
                        args,
                        best_val_metrics,
                        filename="checkpoint_best.pth",
                    )
                    if is_primary():
                        logout(
                            f"Epoch [{curr_epoch}/{args.max_epoch}] "
                            f"saved current best val checkpoint at {filename}; "
                            f"{args.criterion} {eval_metrics[args.criterion]}"
                        )
                        
                # 保存提示词性能统计
                if is_primary():
                    # 从模型中获取提示词性能统计
                    template_performances = None
                    if hasattr(model, 'template_performances'):
                        template_performances = model.template_performances
                    elif hasattr(model, 'module') and hasattr(model.module, 'template_performances'):
                        template_performances = model.module.template_performances
                    
                    if template_performances:
                        # 提取为循环，避免使用lambda
                        model_performance = {}
                        for idx, stats in template_performances.items():
                            avg_cider = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                            model_performance[str(idx)] = {
                                'count': stats['count'],
                                'cider': stats['cider'],
                                'avg_cider': avg_cider
                            }
                        
                        template_stats = {
                            'model_performance': model_performance,
                            'epoch': curr_epoch,
                            'iter': curr_iter
                        }
                        
                        # 从训练数据集中获取提示词性能统计
                        for dataset in dataloaders['train'].dataset.datasets if hasattr(dataloaders['train'].dataset, 'datasets') else [dataloaders['train'].dataset]:
                            if hasattr(dataset, 'task_name') and dataset.task_name == 'scanqa' and hasattr(dataset, 'prompt_performances'):
                                # 同样提取为循环
                                dataset_performance = {}
                                for idx, stats in dataset.prompt_performances.items():
                                    avg_cider = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                                    dataset_performance[str(idx)] = {
                                        'count': stats['count'],
                                        'cider': stats['cider'],
                                        'avg_cider': avg_cider
                                    }
                                template_stats['dataset_performance'] = dataset_performance
                                template_stats['best_prompt_idx'] = dataset.best_prompt_idx
                                break
                        
                        # 保存统计信息
                        template_stats_file = os.path.join(
                            args.checkpoint_dir,
                            f"template_performances_epoch{curr_epoch}_iter{curr_iter}.json"
                        )
                        with open(template_stats_file, "w") as f:
                            json.dump(template_stats, f, indent=4)
                        
                        logout(f"Saved template performance statistics to {template_stats_file}")
            # end of an iteration

        # end of an epoch
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            curr_epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

    # end of training
    eval_metrics = {}
    model.eval()
    for test_loader in dataloaders['test']:
        task_metrics = test_loader.dataset.eval_func(
            args,
            curr_epoch,
            model,
            dataset_config,
            test_loader,
            logout,
            curr_train_iter=curr_iter
        )
        eval_metrics.update(task_metrics)

    # 在评估后打印提示词性能信息
    if is_primary():
        # 打印模型的提示词性能
        if hasattr(model, 'template_performances'):
            template_performances = model.template_performances
            logout("\nModel Prompt Performance Statistics:")
            for idx, stats in sorted(template_performances.items()):
                avg_score = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                logout(f"Prompt {idx}: Avg CIDEr = {avg_score:.4f} (used {stats['count']} times)")
            best_idx = model.best_template_idx if hasattr(model, 'best_template_idx') else None
            if best_idx is not None:
                logout(f"Current Best Prompt Index in Model: {best_idx}\n")
        elif hasattr(model, 'module') and hasattr(model.module, 'template_performances'):
            template_performances = model.module.template_performances
            logout("\nModel Prompt Performance Statistics:")
            for idx, stats in sorted(template_performances.items()):
                avg_score = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                logout(f"Prompt {idx}: Avg CIDEr = {avg_score:.4f} (used {stats['count']} times)")
            best_idx = model.module.best_template_idx if hasattr(model.module, 'best_template_idx') else None
            if best_idx is not None:
                logout(f"Current Best Prompt Index in Model: {best_idx}\n")
                
        # 打印数据集的提示词性能
        for dataset in dataloaders['train'].dataset.datasets if hasattr(dataloaders['train'].dataset, 'datasets') else [dataloaders['train'].dataset]:
            if hasattr(dataset, 'task_name') and hasattr(dataset, 'prompt_performances'):
                if dataset.task_name in ['scanqa', 'scanrefer', 'nr3d']:  # 支持QA、Dense Caption和NR3D任务
                    logout(f"\n{dataset.task_name.upper()} Dataset Prompt Performance Statistics:")
                    for idx, stats in sorted(dataset.prompt_performances.items()):
                        avg_score = stats['cider'] / stats['count'] if stats['count'] > 0 else 0
                        logout(f"Prompt {idx}: Avg CIDEr = {avg_score:.4f} (used {stats['count']} times)")
                    logout(f"Current Best Prompt Index in {dataset.task_name.upper()} Dataset: {dataset.best_prompt_idx}\n")
                break

    return
