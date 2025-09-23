import os
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import *
from losses import  ArcFace, CosFace, PowerFace,SphereFace#,ArcFace_dd,CosFace_dd,SphereFace_dd,PowerFace_dd,ArcFace_d,CosFace_d,SphereFace_d,PowerFace_d
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import PartialFC_V2, my_PFC, LoraFC, my_CE,my_BCE,my_BCE1
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


def evaluate_metrics_torch(backbone, teacher_model, dataloader, device):
    backbone.eval()
    teacher_model.eval()
    
    student_features = []
    teacher_features = []
    
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            student_feat = backbone(img)
            teacher_feat = teacher_model(img)
            
            student_features.append(student_feat)
            teacher_features.append(teacher_feat)
    
    # 合并所有 batch 的特征张量
    student_features = torch.cat(student_features, dim=0)  # [N, D]
    teacher_features = torch.cat(teacher_features, dim=0)  # [N, D]
    
    # 计算余弦相似度（PyTorch 内置函数）
    cos_sims = F.cosine_similarity(student_features, teacher_features, dim=1)  # [N]
    avg_cos_sim = cos_sims.mean().item()  # 标量
    
    # 计算模长比例（学生/教师）
    student_norms = torch.norm(student_features, p=2, dim=1)  # L2 范数 [N]
    teacher_norms = torch.norm(teacher_features, p=2, dim=1)  # [N]
    norm_ratios = student_norms / teacher_norms
    # median_ratio = torch.median(norm_ratios).item()  # 中值
    median_ratio=norm_ratios.mean().item()

    return avg_cos_sim, median_ratio


def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    # summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    # train_set = ImageFolder(cfg.rec, transform)
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)


    teacher_model = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, 
                            num_features=cfg.embedding_size)
    teacher_model.load_state_dict(torch.load(os.path.join(cfg.teacher_ouput, "model.pt")))
    teacher_model.eval().to(device)
    teacher_model.requires_grad_(False)




    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.eval().to(device)
    backbone.load_state_dict(torch.load(os.path.join(cfg.ouput, "model.pt")))

    avg_cos, median_ratio = evaluate_metrics_torch(backbone, teacher_model, train_loader, device)
    print(f"Average Cosine Similarity: {avg_cos:.4f}")
    print(f"Median Norm Ratio (Student/Teacher): {median_ratio:.4f}")
    
    # 记录到日志文件
    with open(os.path.join(cfg.output, "metrics.txt"), "w") as f:
        f.write(f"Cosine Similarity (torch): {avg_cos:.4f}\n")
        f.write(f"Norm Ratio (torch): {median_ratio:.4f}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="casia_teacher", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3_teacher", help='the name of config file')
    # parser.add_argument('--config', default="webface4m", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
