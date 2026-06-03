# --------------------------------------------------------
# MTLoRA with TDCO (Teacher-Distilled Conflict-Aware Optimization)
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from mtl_loss_schemes import MultiTaskLoss, get_loss, DistillationLoss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from ptflops import get_model_complexity_info
from models.lora import mark_only_lora_as_trainable

# Import CAGC
from cagc_processor import CAGCProcessor, should_apply_cagc

try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb library not found. Logging is disabled.")
    wandb_available = False


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=10,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    # distributed training
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='List of tasks to run in MTL setup.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--decoder_map', type=str,
                        help='Path to JSON file containing the type of decoder heads')
    parser.add_argument('--skip_decoder', action='store_true',
                        help='Skip loading decoder head weights')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging.')
    parser.add_argument('--run_name', type=str,
                        help='wandb run name')
    parser.add_argument('--no_eval_50', action='store_true',
                        help='Disable the initial eval at 50 epochs.')

    # Distillation Config
    parser.add_argument('--distill', action='store_true',
                        help='Enable knowledge distillation from teacher models')
    parser.add_argument('--teacher-paths', type=str,
                        help='JSON file mapping tasks to teacher model paths')
    parser.add_argument('--distill-temp', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--distill-alpha', type=float, default=0.5,
                        help='Distillation loss weight')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def load_teacher_models(config, logger):
    """Load multi-teacher models for distillation."""
    teachers = {}
    if not hasattr(config, 'DISTILLATION') or not config.DISTILLATION.ENABLED:
        return teachers

    logger.info("=" * 50)
    logger.info("Loading teacher models for distillation...")
    logger.info(f"Tasks: {config.TASKS}")

    for task in config.TASKS:
        teacher_path = config.MODEL.TEACHER_MODELS.get(task, None)
        if not teacher_path or not os.path.exists(teacher_path):
            logger.warning(f"Teacher model not found for task '{task}': {teacher_path}")
            continue

        # Create single-task teacher model
        teacher_config = get_config_for_task(task)
        teacher_model = build_model(teacher_config)
        teacher_model = build_mtl_model(teacher_model, teacher_config, single_task=task)

        # Load checkpoint
        checkpoint = torch.load(teacher_path, map_location='cpu')
        msg = teacher_model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"Teacher [{task}] loaded from {teacher_path}")
        logger.info(f"Load message: {msg}")

        # Move to GPU and set to eval mode
        teacher_model.cuda()
        teacher_model.eval()

        # Freeze all parameters
        for param in teacher_model.parameters():
            param.requires_grad = False

        teachers[task] = teacher_model

        # Clean up memory
        del checkpoint
        torch.cuda.empty_cache()

    logger.info(f"Successfully loaded {len(teachers)} teacher models")
    logger.info("=" * 50)

    return teachers


def get_config_for_task(task_name, base_config=None):
    """Create config for single-task teacher model."""
    if base_config is None:
        from config import _C
        config = _C.clone()
        config.defrost()
    else:
        config = base_config.clone()
        config.defrost()

    # Set single task
    config.TASKS = [task_name]
    config.MTL = True
    config.MODEL.MTLORA.ENABLED = False  # Teacher models do not use LoRA

    # Set output path
    config.OUTPUT = os.path.join(config.OUTPUT, f"teacher_{task_name}")
    config.freeze()

    return config


def main(config, no_eval_50=False):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    # Build student model
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters / 1e6} M")

    model.cuda()
    macs, params = get_model_complexity_info(model, (3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)

    logger.info(f"ptflops GMACS = {macs} and params = {params}")

    model_without_ddp = model

    # Load teacher models if distillation is enabled
    teachers = {}
    if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED:
        teachers = load_teacher_models(config, logger)
        if len(teachers) != len(config.TASKS):
            logger.warning("Not all tasks have teacher models!")

    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS})
        all_loss_weights = {
            'depth': 1.0,
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights[t]

        criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

        # Distillation loss
        if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers:
            distill_criterion = DistillationLoss(
                temperature=config.DISTILLATION.TEMPERATURE,
                alpha=config.DISTILLATION.ALPHA,
                task_weights=config.DISTILLATION.TASK_WEIGHTS
            )
        else:
            distill_criterion = None
    else:
        distill_criterion = None

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model, 0, teachers=teachers)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model_without_ddp.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if config.EVAL_MODE:
            validate(config, data_loader_val, model, 0, teachers=teachers)
            return

    if config.EVAL_MODE:
        validate(config, data_loader_val, model, 0, teachers=teachers)
        return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model, 0, teachers=teachers)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.MODEL.MTLORA.ENABLED:
        if config.MODEL.MTLORA.FREEZE_PRETRAINED:
            print("\nMarking LoRA params only as trainable:")
            mark_only_lora_as_trainable(model.backbone,
                                        bias=config.MODEL.MTLORA.BIAS,
                                        freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
                                        freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
                                        free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
                                        freeze_downsample_reduction=True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION)
        else:
            print("Marking all layers as trainable")

    if config.MODEL.FREEZE_BACKBONE:
        assert (not config.MODEL.MTLORA.ENABLED)
        print("Freezing backbone.........")
        model.freeze_backbone()

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters()
                      if p.requires_grad and 'lora' in name)
    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_params_without_lora = total_model_params - lora_params
    decoder_params = sum(p.numel() for name, p in model.named_parameters()
                         if 'backbone' not in name)

    print(f"""
Number of trainable params: {trainable_params:,}
Decoder params:             {decoder_params:,}
LoRA params:                {lora_params:,}
Extra params:                {(trainable_params - (lora_params + decoder_params)):,}
Total params:               {total_model_params:,} (trainable ratio: {trainable_params/total_model_params * 100:2.2f}%)
Total params without LoRA:  {total_model_params_without_lora:,} (trainable ratio: {trainable_params/total_model_params_without_lora * 100:2.2f}%)
""")

    # Initialize CAGC
    cagc_processor = None
    if hasattr(config, 'CAGC') and config.CAGC.ENABLED:
        logger.info("[CAGC] Initializing CAGC Processor...")
        cagc_processor = CAGCProcessor(
            tasks=config.TASKS,
            rho_thr=config.CAGC.RHO_THR,
            beta_w=config.CAGC.BETA_W,
            lambda_lap=config.CAGC.LAMBDA_LAP,
            theta_scale=config.CAGC.THETA_SCALE,
            epsilon=config.CAGC.EPSILON if hasattr(config.CAGC, 'EPSILON') else 1e-8,
            device='cuda'
        )
        # Register TA-LoRA parameters from backbone
        cagc_processor.register_shared_params(model.backbone)
        logger.info("[CAGC] CAGC Processor initialized successfully")

    logger.info("Start training")
    start_time = time.perf_counter()

    epoch = 0
    for epoch in range(config.TRAIN.EPOCHS):
        if not config.MTL:
            data_loader_train.sampler.set_epoch(epoch)

        # Use CAGC-enabled training if processor is available
        if cagc_processor is not None:
            train_one_epoch_cagc(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, cagc_processor=cagc_processor, teachers=teachers, distill_criterion=distill_criterion)
        else:
            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, teachers=teachers, distill_criterion=distill_criterion)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)
        if epoch % config.EVAL_FREQ == 0 or (not no_eval_50 and epoch == 50):
            if config.MTL:
                eval_results = validate(config, data_loader_val, model, epoch, teachers=teachers)
                if 'semseg' in eval_results:
                    current_acc = eval_results['semseg']['mIoU']
                    max_accuracy = max(max_accuracy, current_acc)
            else:
                acc1, _, _ = validate(config, data_loader_val, model, epoch, teachers=teachers)
                max_accuracy = max(max_accuracy, acc1)

    # final eval
    validate(config, data_loader_val, model, epoch, teachers=teachers)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info(f'Best accuracy: {max_accuracy:.2f}%')


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, 
                    lr_scheduler, loss_scaler, task=None, teachers=None, distill_criterion=None):
    """Standard training function without CAGC, with optional distillation."""
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    task_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter() if teachers else None
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        if not config.MTL:
            samples, targets = batch
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            samples = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(
                non_blocking=True) for task in config.TASKS}

        # Mixup (not recommended with distillation)
        if mixup_fn is not None and not (hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED):
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

            # Compute task loss
            if not config.MTL:
                task_loss = criterion(outputs, targets)
            else:
                task_loss, loss_dict = criterion(outputs, targets)

            total_loss = task_loss

            # Compute distillation loss if enabled
            if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers and distill_criterion:
                teacher_outputs = {}
                with torch.no_grad():
                    for task_name, teacher in teachers.items():
                        teacher_outputs[task_name] = teacher(samples)

                total_loss, distill_losses = distill_criterion(
                    outputs, teacher_outputs, targets, loss_dict
                )
                distill_loss = sum(distill_losses.values()) if distill_losses else 0
            else:
                distill_loss = 0

        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        if not config.MTL:
            loss_meter.update(total_loss.item(), targets.size(0))
        else:
            loss_meter.update(total_loss.item())
        task_loss_meter.update(task_loss.item())
        if distill_loss_meter and distill_loss > 0:
            distill_loss_meter.update(distill_loss)

        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if wandb_available and dist.get_rank() == 0:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/weight_decay": optimizer.param_groups[0]['weight_decay'],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/grad_norm": norm_meter.val,
                "train/grad_norm_avg": norm_meter.avg,
                "train/loss_scale": scaler_meter.val,
                "train/loss_scale_avg": scaler_meter.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"train/tasks/{task}/loss"] = task_loss.item()
            if distill_loss_meter and distill_loss > 0:
                metrics["train/loss_distill"] = distill_loss_meter.val
                metrics["train/loss_distill_avg"] = distill_loss_meter.avg
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            log_msg = (
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'task_loss {task_loss_meter.val:.4f} ({task_loss_meter.avg:.4f})\t'
            )
            if distill_loss_meter and distill_loss > 0:
                log_msg += f'distill {distill_loss_meter.val:.4f} ({distill_loss_meter.avg:.4f})\t'
            log_msg += (
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB'
            )
            logger.info(log_msg)

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        scores = performance_meter.get_score(verbose=True)
        if wandb_available and dist.get_rank() == 0:
            scores_logs = {
                "train/epoch": epoch,
            }
            if 'semseg' in scores:
                scores_logs["train/tasks/semseg/mIoU"] = scores['semseg']['mIoU']
            if 'normals' in loss_dict:
                scores_logs["train/tasks/normals/mean"] = scores['normals']['mean']
                scores_logs["train/tasks/normals/rmse"] = scores['normals']['rmse']
                scores_logs["train/tasks/normals/mean_v2"] = scores['normals']['mean_v2']
                scores_logs["train/tasks/normals/rmse_v2"] = scores['normals']['rmse_v2']
            if 'human_parts' in loss_dict:
                scores_logs["train/tasks/human_parts/mIoU"] = scores['human_parts']['mIoU']
            if 'sal' in loss_dict:
                scores_logs["train/tasks/sal/maxF"] = scores['sal']['maxF']
                scores_logs["train/tasks/sal/Beta maxF"] = scores['sal']['Beta maxF']
                scores_logs["train/tasks/sal/mIoU"] = scores['sal']['mIoU']
            if 'edge' in loss_dict:
                scores_logs["train/tasks/sal/loss"] = scores['edge']['loss']
            if 'depth' in loss_dict:
                scores_logs["train/tasks/depth/rmse"] = scores['depth']['rmse']
                scores_logs["train/tasks/depth/log_rmse"] = scores['depth']['log_rmse']

            wandb.log(scores_logs)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def train_one_epoch_cagc(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, 
                        lr_scheduler, loss_scaler, cagc_processor=None, teachers=None, distill_criterion=None):
    """
    CAGC-enabled training function with optional distillation.
    Separately computes task gradients on TA-LoRA parameters and coordinates them using CAGC.
    """
    model.train()
    optimizer.zero_grad()

    # Clear CAGC shared gradients
    if cagc_processor is not None:
        cagc_processor.zero_shared_grads()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    task_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter() if teachers else None
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)

    start = time.perf_counter()
    end = time.perf_counter()
    loss_dict = None

    for idx, batch in enumerate(data_loader):
        samples = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in config.TASKS}

        # Mixup (not recommended with distillation)
        if mixup_fn is not None and not (hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED):
            samples, targets = mixup_fn(samples, targets)

        # Check if we should apply CAGC for this step
        use_cagc = (cagc_processor is not None and 
                   should_apply_cagc(epoch, idx, 
                                   config.CAGC.APPLY_FROM_EPOCH if hasattr(config.CAGC, 'APPLY_FROM_EPOCH') else 0,
                                   1))

        if use_cagc:
            # === CAGC Mode ===

            # 1. Compute gradients for each task separately on TA-LoRA parameters
            task_grads = {}
            task_losses = {}

            for task in config.TASKS:
                # Clear gradients
                model.zero_grad(set_to_none=True)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                    outputs = model(samples)
                    if isinstance(outputs, dict):
                        task_output = outputs[task]
                    else:
                        raise TypeError(f"Unexpected output type: {type(outputs)}")

                    task_target = targets[task]

                    # Compute task-specific loss
                    if hasattr(criterion, 'loss_ft'):
                        loss = criterion.loss_ft[task](task_output, task_target)
                    else:
                        loss = criterion(outputs, {task: task_target})[0]

                task_losses[task] = loss.item()

                # Backward pass
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                       parameters=model.parameters(), create_graph=is_second_order,
                                       update_grad=False)

                # Collect TA-LoRA gradients
                grads = []
                for param in cagc_processor.shared_params:
                    if param.grad is not None:
                        grads.append(param.grad.clone().detach().flatten())
                    else:
                        grads.append(torch.zeros_like(param.data).flatten())

                task_grads[task] = grads

            # 2. Apply CAGC to coordinate gradients
            final_flat_grad = cagc_processor.apply_cagc(task_grads)

            # 3. Compute full forward pass for total loss (for logging and TS-LoRA/Decoder gradients)
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)

                # Compute task loss
                task_loss, loss_dict = criterion(outputs, targets)
                total_loss = task_loss

                # Compute distillation loss if enabled
                if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers and distill_criterion:
                    teacher_outputs = {}
                    with torch.no_grad():
                        for task_name, teacher in teachers.items():
                            teacher_outputs[task_name] = teacher(samples)

                    total_loss, distill_losses = distill_criterion(
                        outputs, teacher_outputs, targets, loss_dict
                    )
                    distill_loss = sum(distill_losses.values()) if distill_losses else 0
                else:
                    distill_loss = 0

            # 4. Backward pass for TS-LoRA and Decoder gradients
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                   parameters=model.parameters(), create_graph=is_second_order,
                                   update_grad=False)

            # 5. Override TA-LoRA gradients with CAGC-coordinated gradients
            if loss_scaler is not None:
                scale = loss_scaler.state_dict()["scale"]
                cagc_processor.apply_to_model(final_flat_grad / scale)

            # 6. Optimizer step
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if cagc_processor is not None:
                    cagc_processor.zero_shared_grads()
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

            loss_scale_value = loss_scaler.state_dict()["scale"]

        else:
            # === Standard Mode (no CAGC) ===
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)

                # Compute task loss
                task_loss, loss_dict = criterion(outputs, targets)
                total_loss = task_loss

                # Compute distillation loss if enabled
                if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers and distill_criterion:
                    teacher_outputs = {}
                    with torch.no_grad():
                        for task_name, teacher in teachers.items():
                            teacher_outputs[task_name] = teacher(samples)

                    total_loss, distill_losses = distill_criterion(
                        outputs, teacher_outputs, targets, loss_dict
                    )
                    distill_loss = sum(distill_losses.values()) if distill_losses else 0
                else:
                    distill_loss = 0

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                   parameters=model.parameters(), create_graph=is_second_order,
                                   update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                if cagc_processor is not None:
                    cagc_processor.zero_shared_grads()
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)

            loss_scale_value = loss_scaler.state_dict()["scale"]

        # Logging
        loss_meter.update(total_loss.item())
        task_loss_meter.update(task_loss.item())
        if distill_loss_meter and distill_loss > 0:
            distill_loss_meter.update(distill_loss)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if wandb_available and dist.get_rank() == 0:
            metrics = {
                "train/epoch_ndx": epoch,
                "train/batch_ndx": idx,
                "train/train_loss": loss_meter.val,
                "train/train_loss_avg": loss_meter.avg,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/weight_decay": optimizer.param_groups[0]['weight_decay'],
                "train/time": batch_time.val,
                "train/time_avg": batch_time.avg,
                "train/grad_norm": norm_meter.val,
                "train/grad_norm_avg": norm_meter.avg,
                "train/loss_scale": scaler_meter.val,
                "train/loss_scale_avg": scaler_meter.avg,
                "train/memory": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                "train/cagc_enabled": float(use_cagc),
            }
            if loss_dict is not None:
                for task, task_loss in loss_dict.items():
                    metrics[f"train/tasks/{task}/loss"] = task_loss.item()
            if distill_loss_meter and distill_loss > 0:
                metrics["train/loss_distill"] = distill_loss_meter.val
                metrics["train/loss_distill_avg"] = distill_loss_meter.avg
            wandb.log(metrics)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            cagc_status = " [CAGC]" if use_cagc else ""

            log_msg = (
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]{cagc_status}\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'task_loss {task_loss_meter.val:.4f} ({task_loss_meter.avg:.4f})\t'
            )
            if distill_loss_meter and distill_loss > 0:
                log_msg += f'distill {distill_loss_meter.val:.4f} ({distill_loss_meter.avg:.4f})\t'
            log_msg += (
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB'
            )
            logger.info(log_msg)

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        scores = performance_meter.get_score(verbose=True)
        if wandb_available and dist.get_rank() == 0:
            scores_logs = {
                "train/epoch": epoch,
            }
            if 'semseg' in scores:
                scores_logs["train/tasks/semseg/mIoU"] = scores['semseg']['mIoU']
            if 'normals' in loss_dict:
                scores_logs["train/tasks/normals/mean"] = scores['normals']['mean']
                scores_logs["train/tasks/normals/rmse"] = scores['normals']['rmse']
                scores_logs["train/tasks/normals/mean_v2"] = scores['normals']['mean_v2']
                scores_logs["train/tasks/normals/rmse_v2"] = scores['normals']['rmse_v2']
            if 'human_parts' in loss_dict:
                scores_logs["train/tasks/human_parts/mIoU"] = scores['human_parts']['mIoU']
            if 'sal' in loss_dict:
                scores_logs["train/tasks/sal/maxF"] = scores['sal']['maxF']
                scores_logs["train/tasks/sal/Beta maxF"] = scores['sal']['Beta maxF']
                scores_logs["train/tasks/sal/mIoU"] = scores['sal']['mIoU']
            if 'edge' in loss_dict:
                scores_logs["train/tasks/sal/loss"] = scores['edge']['loss']
            if 'depth' in loss_dict:
                scores_logs["train/tasks/depth/rmse"] = scores['depth']['rmse']
                scores_logs["train/tasks/depth/log_rmse"] = scores['depth']['log_rmse']

            wandb.log(scores_logs)

    epoch_time = time.perf_counter() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch, teachers=None):
    """Evaluate model performance."""
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, config.DATA.DBNAME)
    loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter() if teachers else None

    # Loss functions
    loss_ft = torch.nn.ModuleDict({
        task: get_loss(config['TASKS_CONFIG'], task, config) for task in config.TASKS
    })
    loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    task_loss_weights = {t: loss_weights.get(t, 1.0) for t in config.TASKS}
    criterion = MultiTaskLoss(config.TASKS, loss_ft, task_loss_weights)

    # Distillation criterion for validation logging
    if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers:
        distill_criterion = DistillationLoss(
            temperature=config.DISTILLATION.TEMPERATURE,
            alpha=config.DISTILLATION.ALPHA,
            task_weights=config.DISTILLATION.TASK_WEIGHTS
        )
    else:
        distill_criterion = None

    model.eval()
    logger.info("Start validation")
    start = time.perf_counter()

    for i, batch in enumerate(data_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

        # Student forward
        output = model(images)

        # Compute loss
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            loss, loss_dict = criterion(output, targets)

            # Distillation loss (for logging only)
            if hasattr(config, 'DISTILLATION') and config.DISTILLATION.ENABLED and teachers and distill_criterion:
                teacher_outputs = {}
                for task_name, teacher in teachers.items():
                    teacher_outputs[task_name] = teacher(images)

                _, distill_losses = distill_criterion(output, teacher_outputs, targets, loss_dict)
                distill_loss = sum(distill_losses.values()) if distill_losses else 0
                distill_loss_meter.update(distill_loss)

        loss_meter.update(loss.item())

        # Performance evaluation
        processed_output = {t: get_output(output[t], t) for t in tasks}
        performance_meter.update(processed_output, targets)

        # Wandb logging
        if wandb_available and dist.get_rank() == 0 and i % 10 == 0:
            log_dict = {
                "val/epoch": epoch,
                "val/step": i,
                "val/loss": loss_meter.val,
                "val/loss_avg": loss_meter.avg,
            }
            if distill_loss_meter:
                log_dict["val/distill_loss"] = distill_loss_meter.val
                log_dict["val/distill_loss_avg"] = distill_loss_meter.avg
            wandb.log(log_dict)

    # Compute final metrics
    eval_results = performance_meter.get_score(verbose=True)
    logger.info(f"val loss {loss_meter.avg:.4f}")

    if distill_loss_meter:
        logger.info(f"val distill loss {distill_loss_meter.avg:.4f}")

    epoch_time = time.perf_counter() - start
    logger.info(f"Validation takes {datetime.timedelta(seconds=int(epoch_time))}")

    # Wandb detailed metrics
    if wandb_available and dist.get_rank() == 0:
        log_dict = {
            "val/epoch": epoch,
            "val/loss_final": loss_meter.avg,
            "val/time": epoch_time,
        }

        if distill_loss_meter:
            log_dict["val/distill_loss_final"] = distill_loss_meter.avg

        for task_name in eval_results:
            for metric, value in eval_results[task_name].items():
                if isinstance(value, (int, float)):
                    log_dict[f"val/{task_name}/{metric}"] = value

        wandb.log(log_dict)

    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]

        # Warmup
        for _ in range(50):
            model(images)

        # Test throughput
        logger.info(f"throughput averaged with 30 times")
        torch.cuda.synchronize()
        tic1 = time.perf_counter()
        for _ in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.perf_counter()

        throughput_val = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput_val:.2f} images/s")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # Set random seeds
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Linear scale learning rate
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # Create output directory and logger
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)

    # Save config
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

    # Wandb logging
    if args.disable_wandb:
        wandb_available = False
        logger.info("Wandb logging disabled.")
    elif wandb_available and dist.get_rank() == 0:
        try:
            if not os.getenv("WANDB_API_KEY"):
                wandb.login()
            else:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
            config_name = f"{os.path.basename(os.path.dirname(args.cfg))}/{os.path.basename(args.cfg)}"
            wandb.init(project='MTLoRA-TDCO', config=config,
                       name=config_name if not args.run_name else args.run_name)
            wandb.config.update({'args': vars(args)})
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
            wandb_available = False

    main(config, no_eval_50=args.no_eval_50)
