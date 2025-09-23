import time
import os
import sys
import logging
import torch
import pandas as pd

# from torch import distributed

class logger(object):
    def __init__(self, cfg, start_step=0, writer=None):
        self.frequent: int = cfg.frequent
        # self.rank: int = distributed.get_rank()
        # self.world_size: int = distributed.get_world_size()
        self.rank = 0
        self.world_size = 1
        self.time_start = time.time()
        self.total_step: int = cfg.total_step
        self.start_step: int = start_step
        self.batch_size: int = cfg.batch_size
        self.writer = writer

        self.init = False
        self.tic = 0
        
        self.losses = []
        self.lrs = []
        self.loss_crt = 0
        self.loss_sum = 0
        self.loss_cnt = 0
        self.loss_avg = 0
        
        if self.rank == 0:
            log_root = logging.getLogger()
            log_root.setLevel(logging.INFO)
            formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
            handler_file = logging.FileHandler(os.path.join(cfg.output, "training.log"))
            handler_stream = logging.StreamHandler(sys.stdout)
            handler_file.setFormatter(formatter)
            handler_stream.setFormatter(formatter)
            log_root.addHandler(handler_file)
            log_root.addHandler(handler_stream)
            log_root.info('rank_id: %d' % self.rank)
        
        for key in dir(cfg):
            if key.startswith("__"):
                continue
            num_space = 25 - len(key)
            logging.info(": " + key + " " * num_space + str(getattr(cfg, key)))

    def __call__(self,
                 global_step: int,
                 loss: float,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        self.loss_crt = loss
        self.loss_sum += self.loss_crt
        self.loss_cnt += 1
        self.loss_avg = self.loss_sum / self.loss_cnt
        self.losses.append(loss)
        self.lrs.append(learning_rate)
        
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', self.loss_avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, self.loss_avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, self.loss_avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                # print(msg)
                self.loss_crt = 0
                self.loss_sum = 0
                self.loss_cnt = 0
                self.loss_avg = 0
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

    def loss2csv(self, output):
        csv = pd.DataFrame({"loss": self.losses, "lr": self.lrs})
        csv.to_csv(output+"/loss.csv", index=False)