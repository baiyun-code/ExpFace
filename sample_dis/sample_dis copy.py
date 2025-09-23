import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import *
from losses import  ArcFace, CosFace, PowerFace,SphereFace#,ArcFace_d,CosFace_d,SphereFace_d,PowerFace_d,
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
import importlib

import os
import pandas as pd

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    os.makedirs(os.path.join(cfg.output, "sample_logs"), exist_ok=True)

    results_df = pd.DataFrame(columns=["img_path", "class_center", "condition_type"])
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    # train_set = ImageFolder(cfg.rec, transform)
    train_set = my_ImageFolder1(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.to(device)



    #margin_loss = CombinedMarginLoss(64, cfg.margin_list[0], cfg.margin_list[1], cfg.margin_list[2], cfg.interclass_filtering_threshold)
    # margin_loss = ArcFace()
    # margin_loss = ArcFace_d(num_class=cfg.num_classes)
    # margin_loss=CosFace_d(num_class=cfg.num_classes)
    # margin_loss=SphereFace_d(s=32, num_class=cfg.num_classes)
    # margin_loss=PowerFace_d(num_class=cfg.num_classes)
    # margin_loss=SphereFace(s=32, m=1.7)
    margin_loss = CosFace()
    # margin_loss = TanFace()
    # margin_loss = CosSinFace(m=0.4)
    # margin_loss = ArcSinFace()
    # margin_loss = PowerFace(s=32,m=0.5)
    # margin_loss=ArcFace_s(32, 0.5, Forward_s)
    # margin_loss=PowerFace(m=0.7)
    margin1=55
    margin2=80
    epoch=19

    CE_loss = my_CE11(margin_loss, cfg.embedding_size, cfg.num_classes, False,margin1=margin1,margin2=margin2)
    # CE_loss = LoraFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.bottle_neck, False)
    # CE_loss = my_PFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    CE_loss.to(device)
    

    # opt = torch.optim.Adam(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    # opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # lr_scheduler = MHLR(opt, cfg.total_step)
    # lr_scheduler = LinearLR(opt, start_factor=1, end_factor=0, total_iters=cfg.total_step)
    # lr_scheduler = ExponentialLR(opt, gamma=0.9999)

    start_epoch = 0
    global_step = 0
   


    dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"),weights_only=True)
    backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    CE_loss.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
    del dict_checkpoint

    # log = logger(cfg=cfg, start_step= global_step, writer=summary_writer)

    backbone.eval()
    CE_loss.eval()

    # all_result=[]

    target_label = 0 
    # amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    with torch.no_grad():
        # histogram = torch.zeros((36),device=device)
        # sum_theta=torch.zeros((1),device=device)

        for _, (img_paths, img, local_labels) in enumerate(train_loader):
            global_step += 1
            print(f"Batch {global_step}")
            # Check which img_paths contain '0_494034'
            mask = (local_labels == target_label)

            if mask.any():
                print("have")



    # aver_theta=sum_theta/cfg.num_image
    # histogram_list = histogram.tolist()
    # with open(os.path.join(cfg.output, f"epoch5_histogram.txt"), "w") as f:
    #     for count in histogram_list:
    #         f.write(f"{count}\n")
    #     f.write(f"{aver_theta.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="casia_rcd", help='the name of config file')
    # parser.add_argument('--config', default="ms1mv3_a", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
