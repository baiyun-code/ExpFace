# import pickle
import sys
sys.path.insert(0, ".")
from PIL import Image
import logging
import os
from typing import List
# import cv2
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from utils.utils_logging import init_logging

import argparse
from backbones import get_model
import verification
import importlib
# from logger import logger

def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)

@torch.no_grad()
def load_bin(path, image_size):
    issame_list = np.load(os.path.join(path, "pair_label.npy"))
    img_path = os.path.join(path, "imgs")
    
    imgs = []
    for idx in range(len(issame_list)*2):
        # img = cv2.imread(os.path.join(img_path, f"{idx}.jpg"))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(Image.open(os.path.join(img_path, f"{idx}.jpg")))
        imgs.append(img)
    
    # with open(path, 'rb') as f:
    #     bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        img = imgs[idx]
    #     _bin = bins[idx]
    #     img = mx.image.imdecode(_bin)
        # if img.shape[1] != image_size[0]:
        #     img = mx.image.resize_short(img, image_size[0])
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.copy())
        if idx % 1000 == 0:
            print('loading img', idx)
    print(data_list[0].shape)
    return data_list, issame_list

def init_dataset(val_targets, data_dir, image_size):
    for name in val_targets:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

def ver_test(backbone: torch.nn.Module, global_step: int):
    results = []
    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
            ver_list[i], backbone, 10, 10)
        logging.info('[%s][%d]XNorm: %f' % (ver_name_list[i], global_step, xnorm))
        logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], global_step, acc2, std2))

        summary_writer.add_scalar(tag=ver_name_list[i], scalar_value=acc2, global_step=global_step, )

        if acc2 > highest_acc_list[i]:
            highest_acc_list[i] = acc2
        logging.info(
            '[%s][%d]Accuracy-Highest: %1.5f' % (ver_name_list[i], global_step, highest_acc_list[i]))
        results.append(acc2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="webface4m", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3.base.ms1mv3_arcface", help='the name of config file')
    args = parser.parse_args()
    
    # cfg = get_config(args.config)
    config = importlib.import_module("configs."+args.config)
    cfg = config.cfg()
    
    rec_prefix = cfg.val
    model_path = cfg.output + "/model.pt"
    val_targets = cfg.val_targets
    network = cfg.network
    image_size = cfg.image_size
    embedding_size = cfg.embedding_size
    
    init_logging(0, cfg.output)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    # log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    
    backbone = get_model(network, dropout=0, fp16=False).cuda()
    backbone.load_state_dict(torch.load(model_path))
    backbone = torch.nn.DataParallel(backbone)
    
    highest_acc_list: List[float] = [0.0] * len(val_targets)
    ver_list: List[object] = []
    ver_name_list: List[str] = []
    init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
    
    backbone.eval()
    ver_test(backbone, 6666)
    # backbone.train()