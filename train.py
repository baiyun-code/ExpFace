import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import *
from losses import  ArcFace, CosFace, PowerFace,SphereFace,ArcFace_norm,PowerFace_norm,PowerFace_norm1
#,ArcFace_dd,CosFace_dd,SphereFace_dd,PowerFace_dd,ArcFace_d,CosFace_d,SphereFace_d,PowerFace_d
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import PartialFC_V2, my_PFC, LoraFC, my_CE,my_BCE,my_BCE1,my_CE_logexp1,my_CE_logexp,my_CE_logexp2,Unified_Cross_Entropy_Loss_CosFace,Unified_Cross_Entropy_Loss_ExpFace,MagLinear,CurricularFace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
import importlib

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

def ce(loss_name="my_CE",margin_loss=ArcFace, embedding_size=512, num_classes=1000, sample_rate=1.0,fp16=False,device=None):
    if loss_name == "my_CE":
        return my_CE(margin_loss, embedding_size, num_classes, fp16)
    elif loss_name =="PFC":
        return my_PFC(margin_loss, embedding_size, num_classes, sample_rate, fp16,device)
    elif loss_name == "my_CE_logexp":
        return my_CE_logexp(margin_loss, embedding_size, num_classes, fp16)
    elif loss_name == "my_CE_logexp1":
        return my_CE_logexp1(margin_loss, embedding_size, num_classes, fp16)
    elif loss_name == "my_CE_logexp2":
        return my_CE_logexp2(margin_loss, embedding_size, num_classes, fp16)
    elif loss_name =="Unified_Cross_Entropy_Loss_CosFace":
        return Unified_Cross_Entropy_Loss_CosFace(embedding_size,num_classes,num_classes,fp16)#不重采样不重加权的情况下
    elif loss_name =="Unified_Cross_Entropy_Loss_ExpFace":
        return Unified_Cross_Entropy_Loss_ExpFace(embedding_size,num_classes,num_classes,fp16)#不重采样不重加权的情况下
    elif loss_name=="MagFace":
        return MagLinear(embedding_size, num_classes)
    elif loss_name=="CurricularFace":
        return CurricularFace(embedding_size, num_classes, s=64.0, m=0.5)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    # train_set = ImageFolder(cfg.rec, transform)
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.train().to(device)

    # margin_loss = ArcFace()



    # margin_loss=SphereFace(s=cfg.s, m=cfg.m)
    margin_loss = cfg.margin_loss
    # margin_loss=ArcFace(s=cfg.s,m=cfg.m)
    # margin_loss = PowerFace(s=cfg.s,m=cfg.m)

    # margin_loss=ArcFace_norm(s=cfg.s)
    # margin_loss=PowerFace_norm1(s=cfg.s,m=0.58)

    CE_loss = ce(cfg.loss_name, margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False, cfg.device)
    # CE_loss = my_CE_logexp1(margin_loss, cfg.embedding_size, cfg.num_classes, False)
    # CE_loss1=my_CE(margin_loss1, cfg.embedding_size, cfg.num_classes, False)
    # CE_loss = LoraFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.bottle_neck, False)
    # CE_loss = my_PFC(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, False)
    CE_loss.train().to(device)
    # CE_loss1.eval().to(device)
    
    opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    # opt = torch.optim.Adam(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    # opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # lr_scheduler = MHLR(opt, cfg.total_step)
    # lr_scheduler = LinearLR(opt, start_factor=1, end_factor=0, total_iters=cfg.total_step)
    # lr_scheduler = ExponentialLR(opt, gamma=0.9999)
    lr_scheduler = CosineAnnealingLR(opt, T_max=cfg.total_step)

    start_epoch = 0
    global_step = 0

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{cfg.resume_epoch}.pt"),weights_only=True)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        CE_loss.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
    log = logger(cfg=cfg, start_step= global_step, writer=summary_writer)

    amp = torch.amp.grad_scaler.GradScaler("cuda", growth_interval=100)


    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img.to(device))
            loss = CE_loss(local_embeddings, local_labels.to(device))
            # loss1=CE_loss1(local_embeddings, local_labels.to(device))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            # lr_scheduler.my_step(loss.item())
            lr_scheduler.step()

            with torch.no_grad():
                log(global_step, loss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # if global_step % cfg.verbose == 0 and global_step > 0:
                #     callback_verification(global_step, backbone)



        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": CE_loss.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"))
            del checkpoint

        # path_module = os.path.join(cfg.output, "model.pt")
        # torch.save(backbone.state_dict(), path_module)

    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    path_module = os.path.join(cfg.output, "CE.pt")
    torch.save(CE_loss.state_dict(), path_module)
    log.loss2csv(cfg.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="casia_rcd", help='the name of config file')
    parser.add_argument('--config', default="casia", help='the name of config file')
    # parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    # parser.add_argument('--config', default="webface4m", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
