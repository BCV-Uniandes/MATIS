#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import itertools
import numpy as np
import shutil
import os
import pprint
import random
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats


import matis.models.losses as losses
import matis.models.optimizer as optim
import matis.utils.checkpoint as cu
import matis.utils.distributed as du
import matis.utils.logging as logging
import matis.utils.metrics as metrics
import matis.utils.misc as misc

from matis.datasets import loader
from matis.models import build_model
from matis.utils.meters import EpochTimer, SurgeryMeter
from matis.utils.multigrid import MultigridSchedule
from torch.nn.modules.distance import PairwiseDistance

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    type=torch.float32,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            matis/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    complete_tasks = cfg.TASKS.TASKS
    complete_loss_funs = cfg.TASKS.LOSS_FUNC
    
    for cur_iter, (inputs, labels, _, boxes, ori_boxes, boxes_mask, faster_ftrs, _) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(type).cuda(non_blocking=True)
            else:
                inputs = inputs.to(type).cuda(non_blocking=True)
            
            boxes = boxes.cuda(non_blocking=True)
            ori_boxes = ori_boxes.cuda(non_blocking=True)
            boxes_mask = boxes_mask.cuda(non_blocking=True)
            faster_ftrs = faster_ftrs.to(type).cuda(non_blocking=True) if faster_ftrs is not None else None
            for task in complete_tasks:
                labels[task] = labels[task].to(type).cuda(non_blocking=True)
                
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs, boxes, faster_ftrs, boxes_mask)
            
            # Explicitly declare reduction to mean and compute the loss for each task.
            loss = []
            for idx, task in enumerate(complete_tasks):   
                if task in ['tools','segmentation']:
                    loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                    
                    if 'bce' in complete_loss_funs[idx]:
                        new_labels = torch.nn.functional.one_hot(labels[task],num_classes=cfg.TASKS.NUM_CLASSES[-1]).to(type).cuda(non_blocking=True)
                        tools_loss = loss_fun(preds[task][0], new_labels)
                        loss.append(tools_loss)      
                    else:
                        tools_loss = loss_fun(preds[task][0], labels[task].long())
                        loss.append(tools_loss)
                    
                elif task=='recognition':
                    loss_fun = losses.get_loss_func(complete_loss_funs[idx])(reduction="mean")
                    if 'bce' in complete_loss_funs[idx]:
                        tools_loss = loss_fun(preds[task][0], labels[task].to(type))
                        loss.append(tools_loss)
                    else:
                        loss.append(loss_fun(preds[task][0], labels[task].to(type)))

        if len(complete_tasks) >1:
            final_loss = losses.compute_weighted_loss(loss, cfg.TASKS.LOSS_WEIGHTS)
        else:
            final_loss = loss[0]
            
        # check Nan Loss.
        misc.check_nan_losses(final_loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.NUM_GPUS > 1:
            final_loss = du.all_reduce([final_loss])[0]
        final_loss = final_loss.item()

        # Update and log stats.
        train_meter.update_stats(None, None, None, None, final_loss, loss, lr)
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Train/loss": final_loss, "Train/lr": lr},
                global_step=data_size * cur_epoch + cur_iter,
            )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, type=torch.float32, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            matis/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    # breakpoint()
    for cur_iter, (inputs, labels, image_names, boxes, ori_boxes, boxes_mask, faster_ftrs, ori_idxs) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(type).cuda(non_blocking=True)
            else:
                inputs = inputs.to(type).cuda(non_blocking=True)
            
            boxes = boxes.cuda(non_blocking=True)
            ori_boxes = ori_boxes.cuda(non_blocking=True)
            boxes_mask = boxes_mask.cuda(non_blocking=True)
            faster_ftrs = faster_ftrs.to(type).cuda(non_blocking=True) if faster_ftrs is not None else None
            # masks = masks.to(type).cuda(non_blocking=True) if masks is not None else None
            for task in complete_tasks:
                labels[task] = labels[task].to(type).cuda(non_blocking=True)

                    
        val_meter.data_toc()
        preds = model(inputs, boxes, faster_ftrs, boxes_mask)

        if cfg.NUM_GPUS:
            preds = {task: preds[task][0].cpu() for task in complete_tasks}

        val_meter.iter_toc()
        epoch_names_detect, epoch_bboxes = [], []
        for bid,idx in enumerate(ori_idxs):
            epoch_bboxes.append(ori_boxes[bid].cpu().tolist())
            epoch_names_detect.append(image_names[idx])
        
        assert len(preds['tools'])==len(epoch_names_detect)

        # Images names phases/steps
        epoch_names = image_names
        
        # Update and log stats.
        val_meter.update_stats(preds, epoch_bboxes, epoch_names_detect, epoch_names)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    task_iou, mean_iou, out_files = val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_iou}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )
    val_meter.reset()

    return task_iou, mean_iou, out_files

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            matis/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    # torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))
    # breakpoint()
    # Build the video model and print model statistics.

    type = torch.float32 if cfg.TYPE==32 else torch.float64
    model = build_model(cfg).to(type)
    logger.info(pprint.pformat(model))
        
    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)    
        
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_iou = {task: 0 for task in complete_tasks}
    best_mean_iou = 0
    epoch_timer = EpochTimer()

    if cfg.TEST.ENABLE:
        iou_task, mean_iou, out_files = eval_epoch(val_loader, 
                                                    model, 
                                                    val_meter, 
                                                    0, 
                                                    cfg, 
                                                    type, 
                                                    writer)
        main_path = os.path.split(list(out_files.values())[0])[0]
        fold = main_path.split('/')[-1]
        best_preds_path = main_path.replace(fold, fold+'/best_predictions')
        
        if not cfg.TRAIN.ENABLE:
            return

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()

        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            type,
            writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None 
        )

        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
        if os.path.exists(del_fil):
            os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            iou_task, mean_iou, out_files = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, type, writer)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    logger.info("Best mean iou at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in complete_tasks:
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in complete_tasks:
                    if iou_task[task][0] > best_task_iou[task]:
                        best_task_iou[task] = iou_task[task][0]
                        logger.info("Best {} iou at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
    if writer is not None:
        writer.close()