
import math
from typing import Callable

import torch
from torch import distributed
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.nn.functional import linear, normalize,binary_cross_entropy_with_logits,mse_loss

class my_CE(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.amp.autocast('cuda',enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss

class my_CE_logexp(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE_logexp, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)
        # logits1=logits.detach().clone()
        
        logits2, sin_theta, sin_theta_plus_m, sin_m=self.margin_softmax(logits, labels)
        # logits1=self.margin_softmax(logits1, labels)

        with torch.no_grad():
            batch_size = labels.size(0)
            C,d= norm_weight_activated.shape
            batch_norm_weight_activated=norm_weight_activated.unsqueeze(0).expand(batch_size, -1, -1)
            index = torch.where(labels != -1)[0]
            target_weight=batch_norm_weight_activated[index, labels[index].view(-1)]

            all_labels = torch.arange(C, device=batch_norm_weight_activated.device).expand(batch_size, -1)
            non_target_mask = torch.ones((batch_size, C), dtype=torch.bool, device=batch_norm_weight_activated.device)
            non_target_mask[index, labels[index]] = False  # 目标位置 = False
            nontarget_indices = [all_labels[i][non_target_mask[i]] for i in range(batch_size)]
            nontarget_indices = torch.stack(nontarget_indices)  # (B, C-1)
            non_target_weight = torch.gather(
                batch_norm_weight_activated,
                dim=1,
                index=nontarget_indices.unsqueeze(-1).expand(-1, -1, d)
            )
            weight_m=(-sin_m.unsqueeze(1)*norm_embeddings+sin_theta_plus_m.unsqueeze(1)*target_weight)/sin_theta.unsqueeze(1)
            weight_m=weight_m.unsqueeze(1)
            diff_weight_norm=torch.norm(non_target_weight-weight_m,p=2,dim=-1).detach().clone()#BXC-1

        e_logits=torch.sum(torch.exp(logits2 / diff_weight_norm), dim=1)
        loss=torch.mean(torch.log1p(e_logits),dim=0)



        # loss1= self.cross_entropy(logits1, labels)

        # loss = self.cross_entropy(logits, labels)
        return loss

class my_CE_logexp1(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE_logexp1, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)
        # logits1=logits.detach().clone()
        
        logits, theta_m=self.margin_softmax(logits, labels)
        # logits1=self.margin_softmax(logits1, labels)
        
        with torch.no_grad():
            batch_size = labels.size(0)
            C,d= norm_weight_activated.shape

            index = torch.where(labels != -1)[0]
            target_weight=norm_weight_activated[labels[index].view(-1)]

            #Wj*Wi
            costhetaji = linear(target_weight, norm_weight_activated)
            costhetaji=costhetaji.clamp(-1,1)

            costhetaji.arccos_()
            costhetaji+=theta_m.unsqueeze(1)
            costhetaji.cos_()


            Wd_norm=((((1-costhetaji+ 1e-8)*2)**0.5))
            mask = torch.ones(batch_size, C, dtype=torch.bool)    # 初始化为全 True
            mask[torch.arange(batch_size), labels] = False         # 将 label 对应的位置设为 False
            # 使用掩码筛选元素
            Wd_norm = Wd_norm[mask].view(batch_size, C-1)
            Wd_norm=Wd_norm.detach().clone()

            

        safe_exp_input = torch.clamp(logits / (Wd_norm + 1e-6), max=80)
        e_logits = torch.sum(torch.exp(safe_exp_input), dim=1)
        loss = torch.mean(torch.log1p(e_logits), dim=0)
        # Wd_norm = torch.clamp(Wd_norm, min=1e-3)
        # input_exp = logits / (Wd_norm + 1e-6)
        # max_input = torch.max(input_exp, dim=1, keepdim=True).values
        # stable_exp = torch.exp(input_exp - max_input)
        # e_logits = torch.sum(stable_exp, dim=1) * torch.exp(max_input.squeeze())
        # loss = torch.mean(torch.log(1 + e_logits + 1e-6))

        return loss

class my_CE_logexp2(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE_logexp2, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)
        # logits1=logits.detach().clone()
        
        logits2, sin_theta, sin_theta_plus_m, sin_m=self.margin_softmax(logits, labels)
        # logits1=self.margin_softmax(logits1, labels)

        with torch.no_grad():
            batch_size = labels.size(0)
            C,d= norm_weight_activated.shape
            batch_norm_weight_activated=norm_weight_activated.unsqueeze(0).expand(batch_size, -1, -1)
            index = torch.where(labels != -1)[0]
            target_weight=batch_norm_weight_activated[index, labels[index].view(-1)]

            weight_m=(-sin_m.unsqueeze(1)*norm_embeddings+sin_theta_plus_m.unsqueeze(1)*target_weight)/sin_theta.unsqueeze(1)

            costhetaji = linear(weight_m, norm_weight_activated)
            costhetaji=costhetaji.clamp(-1,1)



            Wd_norm=((((1-costhetaji)*2)**0.5))
            mask = torch.ones(batch_size, C, dtype=torch.bool)    # 初始化为全 True
            mask[torch.arange(batch_size), labels] = False         # 将 label 对应的位置设为 False
            # 使用掩码筛选元素
            Wd_norm = Wd_norm[mask].view(batch_size, C-1)
            Wd_norm=Wd_norm.detach().clone()

        e_logits=torch.sum(torch.exp(logits2 / Wd_norm), dim=1)
        loss=torch.mean(torch.log1p(e_logits),dim=0)



        # loss1= self.cross_entropy(logits1, labels)

        # loss = self.cross_entropy(logits, labels)
        return loss

class my_CE_magnitude(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
        scale_constraint: float = 1.0,  # 目标模长（默认1.0）
        lambda_scale: float = 0.1,
    ):
        super(my_CE_magnitude, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.scale_constraint = scale_constraint
        self.lambda_scale = lambda_scale
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
            raw_norms = torch.norm(embeddings, p=2, dim=1)
            scale_loss = mse_loss(
                raw_norms,
                torch.ones_like(raw_norms) * self.scale_constraint,
                reduction='mean'
            )
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        ce_loss = self.cross_entropy(logits, labels)
        total_loss = ce_loss + self.lambda_scale * scale_loss
        return total_loss

class my_CE11(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
        margin1=65,
        margin2=100
    ):
        super(my_CE11, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.num_classes=num_classes
        self.margin1=margin1
        self.margin2=margin2
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        cos_logits = logits.clamp(-1, 1)

        index = torch.where(labels != -1)[0]
        logits_angles = (cos_logits[index].arccos()/torch.pi)*180 # 创建新张量存储角度
        target_logit_angles = logits_angles[torch.arange(len(index)), labels[index].view(-1)]



        # 将结果转换为列表
        margin1=self.margin1
        margin2=self.margin2

        mask1=target_logit_angles<margin1
        mask2=target_logit_angles>margin2

        non_target_angle_mask1 = torch.ones_like(logits_angles, dtype=torch.bool)
        non_target_angle_mask1[torch.arange(len(index)), labels[index].view(-1)] = False

        result1=[]
        result2=[]
        result_mask1=[]
        result_mask2=[]


        rows1 = torch.where(mask1)[0]#目标角度小于阈值的
        if rows1.numel()!=0:
            # non_target_angles1 = logits_angles[rows1][non_target_angle_mask1[rows1]]
            # targets_angel1 = target_logit_angles[rows1].unsqueeze(1).expand(-1, self.num_classes-1)
            # pairs1 = torch.stack([targets_angel1.flatten(), non_target_angles1.flatten()], dim=1)
            # result_mask1 = pairs1.tolist()#这是只有目标角度满足的部分。

            condition1=(logits_angles<margin1)&non_target_angle_mask1 & mask1.unsqueeze(1)
            row_vail1,cols1=torch.where(condition1)
            row1=index[row_vail1]
            logit_vals1 = logits_angles[row_vail1, cols1]
            target_vals1 = target_logit_angles[row_vail1]

            result1 = list(zip(
            row1.tolist(), 
            cols1.tolist(),
            target_vals1.tolist(),
            logit_vals1.tolist()
            ))


        # ------------------------- Mask2 处理 -------------------------
        rows2 = torch.where(mask2)[0]

        if rows2.numel()!=0:
            # non_target_angles2 = logits_angles[rows2][non_target_mask1[rows2]]
            # targets2 = target_logit_angles[rows2].unsqueeze(1).expand(-1, self.num_classes-1)
            # pairs2 = torch.stack([targets2.flatten(), non_target_angles2.flatten()], dim=1)
            # result_mask2 = pairs2.tolist()




            condition2=(logits_angles>(margin2))&non_target_angle_mask1 & mask2.unsqueeze(1)

            if condition2.any():
                row_vail2,cols2=torch.where(condition2)
                row2=index[row_vail2]
                logit_vals2 = logits_angles[row_vail2, cols2]
                target_vals2 = target_logit_angles[row_vail2]


                # Get top-5 indices (on the original filtered data before zipping)
                topk_values, topk_indices = torch.topk(logit_vals2, k=3)  # k=5
                # Extract only the top-5 using the indices
                row_top5 = row2[topk_indices]
                cols_top5 = cols2[topk_indices]
                target_vals_top5 = target_vals2[topk_indices]
                logit_vals_top5 = logit_vals2[topk_indices]



                result2 = list(zip(
                    row_top5.tolist(),
                    cols_top5.tolist(),
                    target_vals_top5.tolist(),
                    logit_vals_top5.tolist()
                ))
        return result1,result2
    
class my_CE2(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_CE2, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.num_classes=num_classes
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        # target_labels: torch.Tensor,
        non_target_labels: torch.Tensor
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)


        valid_labels = non_target_labels
        
        index = torch.where(labels != -1)[0]

        
        # 批量提取logits
        self_logits = logits[index, labels]
        neighbor_logits = logits[index, valid_labels]
        
        # 计算角度
        result_angles = torch.stack([
            (self_logits.arccos()/torch.pi)*180,
            (neighbor_logits.arccos()/torch.pi)*180
        ], dim=1)
    
        return result_angles  # 形状: [num_valid_samples, 2]

class my_BCE(torch.nn.Module):
    def __init__(
        self,
        s,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_BCE, self).__init__()
        # self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.s=s
        self.embedding_size = embedding_size
        self.num_classes=num_classes
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        

        p_loss = torch.log(1 + torch.exp(-logits.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(logits.clamp(min=-self.s, max=self.s)))



        one_hot = torch.zeros((labels.size(0), self.num_classes), dtype=torch.bool)
        one_hot = one_hot.cuda() if logits.is_cuda else one_hot
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
                # one_hot = torch.index_select(one_hot, 1, partial_index)
        loss1=(~one_hot) * n_loss
        loss1=loss1.sum(dim=1)
        loss1=loss1/(self.num_classes-1)
        loss2=one_hot * p_loss
        loss2=loss2.sum(dim=1)
        loss=loss1+loss2
        # loss = one_hot * p_loss + (~one_hot) * n_loss/(self.num_classes-1)

        return loss.mean()
    
class my_BCE1(torch.nn.Module):
    def __init__(
        self,
        s,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        fp16: bool = False,
    ):
        super(my_BCE1, self).__init__()
        # self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.s=s
        self.embedding_size = embedding_size
        self.num_classes=num_classes
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        

        p_loss = torch.log(1 + torch.exp(-logits.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(logits.clamp(min=-self.s, max=self.s)))



        one_hot = torch.zeros((labels.size(0), self.num_classes), dtype=torch.bool)
        one_hot = one_hot.cuda() if logits.is_cuda else one_hot
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
                # one_hot = torch.index_select(one_hot, 1, partial_index)
        loss1=(~one_hot) * n_loss
        loss1=loss1.sum(dim=1)
        loss1=loss1#/(self.num_classes-1)
        loss2=one_hot * p_loss
        loss2=loss2.sum(dim=1)
        loss=30*loss2+loss1
        # loss = (self.num_classes-1)*one_hot * p_loss + (~one_hot) * n_loss

        return loss.mean()


class my_PFC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
        device=None
    ):
        super(my_PFC, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size)))
        self.num_sample: int = int(sample_rate * num_classes)
        self.classes = torch.arange(num_classes, dtype=torch.int32)
        self.mask = torch.ones(num_classes, dtype=torch.bool)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        if self.sample_rate < 1:
            with torch.no_grad():
                positive, idx_ivs = torch.unique(labels, sorted=False, return_inverse=True)
                
                self.mask[positive] = False
                negative = self.classes[self.mask]
                self.mask[positive] = True
                perm = torch.randperm(negative.shape[0])
                
                index = torch.zeros(self.num_sample, dtype=torch.int32)
                index[:positive.shape[0]] = positive
                index[positive.shape[0]:] = negative[perm[:(self.num_sample - positive.shape[0])]]
                
            weight = self.weight[index]
            labels = idx_ivs
        else:
            weight = self.weight

        with torch.amp.autocast(self.device,enabled=self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
    

class LoraFC(torch.nn.Module):
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        bottle_neck: int,
        fp16: bool = False,
    ):
        super(LoraFC, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.fp16 = fp16
        self.weight0 = torch.nn.Parameter(torch.normal(0, 0.01, (bottle_neck, embedding_size)))
        self.weight1 = torch.nn.Parameter(torch.normal(0, 0.01, (num_classes, bottle_neck)))
        
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        # weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            # norm_embeddings0 = normalize(embeddings)
            # norm_weight_activated0 = normalize(self.weight0)
            # embeddings1 = linear(norm_embeddings0, norm_weight_activated0)
            embeddings1 = linear(embeddings, self.weight0)
            
            norm_embeddings1 = normalize(embeddings1)
            norm_weight_activated1 = normalize(self.weight1)
            logits = linear(norm_embeddings1, norm_weight_activated1)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels)
        return loss
    
class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    def sample(self, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.amp.autocast(self.fp16):
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss

class Unified_Cross_Entropy_Loss_CosFace(nn.Module):
    def __init__(self, in_features, out_features, num_classes: int, fp16: bool = False,m=0.4, s=64, l=1.0, r=1.0):
        super(Unified_Cross_Entropy_Loss_CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))
        
        self.num_class=num_classes
        self.fp16=fp16

    def forward(self, input, label):
        positive = torch.unique(label, sorted=True)
        perm = torch.randperm(self.num_class)
        perm[positive] = 0
        indices = torch.topk(perm, k=int(self.num_class*self.r), largest=False)[1]
        partial_index = indices.sort()[0]
        partial_index = partial_index.cuda()
        
        with torch.amp.autocast('cuda',enabled=self.fp16):
            cos_theta = linear(normalize(input, eps=1e-5), normalize(self.weight[partial_index], eps=1e-5))

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

class Unified_Cross_Entropy_Loss_ExpFace(nn.Module):
    def __init__(self, in_features, out_features, num_classes: int, fp16: bool = False,m=0.69, s=64, l=1.0, r=1.0):
        super(Unified_Cross_Entropy_Loss_ExpFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.l = l
        self.r = r
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*r*10))
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.register_buffer('weight_mom', torch.zeros_like(self.weight))
        
        self.num_class=num_classes
        self.fp16=fp16

    def forward(self, input, label):
        positive = torch.unique(label, sorted=True)
        perm = torch.randperm(self.num_class)
        perm[positive] = 0
        indices = torch.topk(perm, k=int(self.num_class*self.r), largest=False)[1]
        partial_index = indices.sort()[0]
        partial_index = partial_index.cuda()
        
        
        with torch.amp.autocast('cuda',enabled=self.fp16):
            cos_theta = linear(normalize(input, eps=1e-5), normalize(self.weight[partial_index], eps=1e-5))

        index = torch.where(label != -1)[0]
        target_logit = cos_theta[index, label[index].view(-1)]
        
        with torch.no_grad():
            target_logit.arccos_()
            final_target_logit = (target_logit/math.pi).pow(self.m)*math.pi
            final_target_logit.cos_()
            cos_theta[index, label[index].view(-1)] = final_target_logit
        
        cos_m_theta_p = self.s * (cos_theta) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s))) * self.l

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = torch.index_select(one_hot, 1, partial_index)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply


class MagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True, l_margin=0.45, u_margin=0.8, l_a=10,u_a=110, lambda_g=20.0):
        super(MagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.l_a = l_a
        self.u_a = u_a
        self.cut_off = np.cos(np.pi/2-l_margin)
        self.large_value = 1 << 10
        self.lambda_g=lambda_g

    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)
    
    def forward(self, x, target):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        ada_margin = self._margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta
        
        #[cos_theta, cos_theta_m], x_norm
        
        loss_g = self.calc_loss_G(x_norm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        
        loss = loss + self.lambda_g * loss_g

        return loss
    
class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings)
        kernel_norm = F.normalize(self.kernel)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        # with torch.no_grad():
        #     origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        
        loss= self.cross_entropy(output, label)
        return loss
