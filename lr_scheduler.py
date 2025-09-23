from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import SGD
import torch
import warnings
import pandas as pd
import numpy as np

# class MHLR(_LRScheduler):
#     def __init__(self, optimizer, total_steps, alpha=0.001, beta=0.001, lambd=4e-5, tau=0.04, delta=2, last_epoch=-1):
#         self.alpha = alpha
#         self.beta = beta
#         self.toler = int(total_steps*tau)
#         self.thres = lambd
#         self.decay = delta
#         self.weight = 1
#         self.count = 0
        
#         self.L_t = 0
#         self.L_t_1 = 0
#         self.D_t = 0
#         self.D_t_1 = 0
#         self.D_EMA_t = 0
#         self.D_EMA_t_1 = 0
#         # self.k1 = np.ones(self.m)
#         # self.k2 = np.ones(self.m)
#         # self.k2[int(self.m/2):] = -1
#         # self.ss = np.zeros(self.m)
#         # self.avg = []
#         # self.diff = []
#         # self.count = 0
#         # self.weight = 1
#         # self.w_count = 0
#         super().__init__(optimizer, last_epoch=last_epoch)
    
#     def my_step(self, loss):
#         # print(self.last_epoch)
#         # print(self._step_count)
#         super().step()
        
#         t = self.last_epoch - 1
#         if t < 1:
#             self.L_t_1 = loss
#             return
        
#         self.L_t = loss
#         if t < 2:
#             self.D_t = self.alpha*(self.L_t_1 - self.L_t)
#             self.D_EMA_t = self.D_t
#         else:
#             self.D_t = (1 - self.alpha)*self.D_t_1 + self.alpha*(self.L_t_1 - self.L_t)
#             self.D_EMA_t = self.beta*self.D_t + (1-self.beta)*self.D_EMA_t_1
        
#         if self.D_EMA_t < self.thres:
#             if self.count < self.toler:
#                 self.weight = 1
#                 self.count += 1
#             else:
#                 self.weight = 1/self.decay
#                 self.count = 0
#         else:
#             self.weight = 1
#             self.count = 0
                
#         self.L_t_1 = self.L_t
#         self.D_t_1 = self.D_t
#         self.D_EMA_t_1 = self.D_EMA_t
        
#         # return self.weight
#         # if i < self.m:
#         #     self.ss[i] = loss
#         #     return
        
#         # self.ss[:-1] = self.ss[1:]
#         # self.ss[-1] = loss
#         # self.avg.append(np.sum(self.ss*self.k1)/self.m)
        
#         # if i < 2*self.m:
#         #     return
        
#         # s_avg = self.avg[(i-2*self.m):(i-self.m)]
#         # self.diff.append(np.sum(s_avg*self.k2)/self.m)
        
#         # if self.diff[-1] < 0.06:
#         #     self.count += 1
            
#         #     if self.count > 2*self.m:
#         #         self.weight = 0.5
#         #         self.count = 0
#         #         self.w_count += 1
#         #     else:
#         #         self.weight = 1
                
#         #     if self.w_count > 2:
#         #         self.weight = 4
#         #         self.w_count = 0
#         # else:
#         #     self.weight = 1
#         #     self.count = 0
#         #     self.w_count = 0
        
        
        
        
        
        
    
#     def get_lr(self):
#         # if self.last_epoch <= 2*self.m:
#         #     return [0, 0]
        
#         # return [self.weight, self.weight]
        
#         return [(group["lr"]* self.weight) for group in self.optimizer.param_groups]
        
class MHLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, alpha=0.001, beta=0.001, lambd=5e-5, tau=0.05, delta=2, last_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        self.toler = int(total_steps*tau)
        self.thres = lambd
        self.delta = delta
        self.weight = 1
        self.count = 0
        self.power = 0
        
        self.L_t = 0
        self.L_t_1 = 0
        self.D_t = 0
        self.D_t_1 = 0
        self.D_EMA_t = 0
        self.D_EMA_t_1 = 0
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def my_step(self, loss):
        super().step()
        
        t = self.last_epoch - 1
        if t < 1:
            self.L_t_1 = loss
            return
        
        self.L_t = loss
        if t < 2:
            self.D_t = self.alpha*(self.L_t_1 - self.L_t)
            self.D_EMA_t = self.D_t
        else:
            self.D_t = (1 - self.alpha)*self.D_t_1 + self.alpha*(self.L_t_1 - self.L_t)
            self.D_EMA_t = self.beta*self.D_t + (1-self.beta)*self.D_EMA_t_1
        
        if self.D_EMA_t < self.thres:
            if self.count < self.toler:
                self.weight = 1
                self.count += 1
            else:
                self.weight = 1/self.delta
                self.count = 0
                self.power += 1
                
            if self.power > 8:
                self.weight = 1
                # self.weight = self.delta**(self.power-1)
                # self.power = 0
        else:
            self.weight = 1
            self.count = 0
                
        self.L_t_1 = self.L_t
        self.D_t_1 = self.D_t
        self.D_EMA_t_1 = self.D_EMA_t
        
    def get_lr(self):
        return [(group["lr"]*self.weight) for group in self.optimizer.param_groups]


# class PolynomialLRWarmup(_LRScheduler):
#     def __init__(self, optimizer, warmup_iters, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
#         super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
#         self.total_iters = total_iters
#         self.power = power
#         self.warmup_iters = warmup_iters


#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)

#         if self.last_epoch == 0 or self.last_epoch > self.total_iters:
#             return [group["lr"] for group in self.optimizer.param_groups]

#         if self.last_epoch <= self.warmup_iters:
#             return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
#         else:        
#             l = self.last_epoch
#             w = self.warmup_iters
#             t = self.total_iters
#             decay_factor = ((1.0 - (l - w) / (t - w)) / (1.0 - (l - 1 - w) / (t - w))) ** self.power
#         return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

#     def _get_closed_form_lr(self):

#         if self.last_epoch <= self.warmup_iters:
#             return [
#                 base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
#         else:
#             return [
#                 (
#                     base_lr * (1.0 - (min(self.total_iters, self.last_epoch) - self.warmup_iters) / (self.total_iters - self.warmup_iters)) ** self.power
#                 )
#                 for base_lr in self.base_lrs
#             ]

    
if __name__ == "__main__":

    class TestModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 32)
        
        def forward(self, x):
            return self.linear(x)

    test_module = TestModule()
    test_module_pfc = torch.nn.CrossEntropyLoss()
    lr_pfc_weight = 1 / 3
    base_lr = 0.02
    
    sgd = SGD([
        {"params": test_module.parameters(), "lr": base_lr},
        {"params": test_module_pfc.parameters(), "lr": base_lr * lr_pfc_weight}
        ], base_lr)

    x = []
    y = []
    y_pfc = []
    losses = pd.read_csv("Output/wf12m_epoch5_r100_lr0.02~0.000039/loss.csv")["loss"].values
    total_steps = losses.shape[0]
    
    # scheduler = PolynomialLRWarmup(sgd, total_steps//10, total_steps, power=2)
    scheduler = MHLR(sgd, total_steps)
    for i in range(total_steps):
        if i%1000==0:
            print(i)
        
        sgd.step()
        scheduler.my_step(losses[i])
        lr = scheduler.get_last_lr()[0]
        lr_pfc = scheduler.get_last_lr()[1]
        x.append(i)
        y.append(lr)
        y_pfc.append(lr_pfc)

    import matplotlib.pyplot as plt
    fontsize=15
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, linestyle='-', linewidth=2, )
    plt.plot(x, y_pfc, linestyle='-', linewidth=2, )
    plt.xlabel('Iterations')     # x_label
    plt.ylabel("Lr")             # y_label
    # plt.savefig("tmp.png", dpi=600, bbox_inches='tight')
