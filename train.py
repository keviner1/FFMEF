#!/usr/local/bin/python
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import time
import os
import datetime
import random
import numpy as np
from thop import profile
import argparse
import model.FFMEF as FFMEF
from config import config_1
from config import config_2
from config import config_3
from torch.utils.data import DataLoader
from myUtils import build_dataset, train_one_epoch, validate
from pathlib import Path
import myUtils as myUtils
import logging
import math
from torch.cuda.amp import autocast as autocast, GradScaler

def run(args):
    #-----------------------------------log--------------------------------
    myUtils.setup_logger('base', args.SERVER.OUTPUT, level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    # ---------------------------------DDP--------------------------------
    seed = 3407
    rank = 0
    world_size = 1
    args.distributed = False
    local_rank = 0
    logger.info("no distributed")
    # logger.info(f"cur: {torch.cuda.get_device_name(0)}--{local_rank}; cuda: {torch.cuda.is_available()}; gpus: {torch.cuda.device_count()}")


    # device = torch.device("cuda")
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    # ---------------------------------main_model--------------------------------
    if args.MODEL.choice == "FFMEF":
        model = FFMEF.FFMEF(channels=args.MODEL.channels)
    model.cuda(local_rank)

    # ----------------------------criterion-----optimizer-----LR_Schedule------------------------------
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.TRAIN.BASE_LR)
    if args.TRAIN.LR_MODE == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.TRAIN.LR_STEP, gamma=args.TRAIN.LR_DECAY)
    elif args.TRAIN.LR_MODE == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.TRAIN.MILESTONES, gamma=args.TRAIN.LR_DECAY)
    
    # ---------------------------------dataloader--------------------------------
    dataset_train = build_dataset(args, mode='train')
    dataset_val = build_dataset(args, mode='val')
    dataset_val_len = len(dataset_val)
    dataset_train_len = len(dataset_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.TRAIN.BATCH_SIZE, drop_last=True)
    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.TRAIN.NUM_WORKERS, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.TRAIN.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.TRAIN.NUM_WORKERS,
                                pin_memory=True)
    
    # ---------------------------------start--------------------------------
    start_time = time.time()
    start_epoch = 1
    #-----------------------------------params--and---flops-------------------------
    test_RGB = torch.randn(2,3,256,256).cuda()
    test_Y = torch.randn(2,1,256,256).cuda()
    test_Cb = [torch.randn(2,1,256,256).cuda(), torch.randn(2,1,256,256).cuda()]
    test_Cr = [torch.randn(2,1,256,256).cuda(), torch.randn(2,1,256,256).cuda()]
    test_gra = [torch.randn(2,1,256,256).cuda(), torch.randn(2,1,256,256).cuda()]
    model.eval()
    flops, params = profile(model, (test_RGB, test_Y, test_RGB, test_Y))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.distributed == False or torch.distributed.get_rank() == 0:
        logger.info(f'flops: {(flops/1024/1024/1024):.4f}G;  params: {(params/1024/1024):.4f}M')
        logger.info('number of model_main params: %.4f M' % (n_parameters / 1024 / 1024))
        logger.info(f"|--start training--NC:{args.MODEL.channels}--BS:{args.TRAIN.BATCH_SIZE}--LR:{args.TRAIN.BASE_LR}--LS:{args.TRAIN.LR_STEP}|")
    
    # ---------------------------------RESUME--------------------------------
    if args.TRAIN.RESUME != "":
        checkpoint = torch.load(args.TRAIN.RESUME)
        net = checkpoint['model']
        net = {key.replace("module.", ""): val for key, val in net.items()}
        if args.distributed == False or torch.distributed.get_rank() == 0:
            logger.info('resume from %s' % args.TRAIN.RESUME)
        model.load_state_dict(net,strict=False)
        start_epoch = checkpoint['epoch']+1

    #------------------------------------amp------------------------------------
    if args.TRAIN.amp == True:
        scaler = GradScaler()
    else:
        scaler = None  
    # ---------------------------------Train and Validate--------------------------------
    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        #train
        train_one_epoch(args, model, dataloader_train, optimizer, lr_scheduler, epoch, logger, scaler)
        #validation
        eval_status = validate(args, model, dataloader_val, epoch, logger, dataset_val_len)
        Path(args.SERVER.OUTPUT+"/checkpoints").mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(args.SERVER.OUTPUT+f"/checkpoints", f'checkpoint_{epoch:04}.pth')
        best_checkpoint = {
            'model': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            # 'args': args,
        }
        torch.save(best_checkpoint, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.distributed == False or  torch.distributed.get_rank() == 0:
        Logger_msg = "---------------Training time {}".format(total_time_str)
        logger.info(Logger_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default = -1)
    parser.add_argument("--config", default = 1, type=int)
    args = parser.parse_args()
    if args.config == 1:
        run(config_1.get_config())
    if args.config == 2:
        run(config_2.get_config())
    if args.config == 3:
        run(config_3.get_config())
