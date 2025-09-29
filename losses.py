from typing import Any
import torch
import math
import torch.nn.functional as F

class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0] # Why there are -1 in labels and we must ignore those -1?
        # index_positive = labels

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.m
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
    

    
class SphereFace(torch.nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, s=30., m=1.5):
        super(SphereFace, self).__init__()
        # self.feat_dim = feat_dim
        # self.num_class = num_class
        self.s = s
        self.m = m
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        # with torch.no_grad():
            # self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        # cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            m_theta = torch.acos(x)
            m_theta.scatter_(
                1, y.view(-1, 1), self.m, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - x

        logits = self.s * (x + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits

class ArcFace_norm(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace_norm, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        batch_size = index.size(0)
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            theta=target_logit.arccos()
            theta_plus_m=theta + self.m
            final_target_logit = (theta_plus_m).cos()
            logits[index, labels[index].view(-1)] = final_target_logit

            sin_theta = theta.sin()                            # sin(θ)
            sin_theta_plus_m = theta_plus_m.sin()          # sin(θ + m)
            sin_m =torch.tensor( math.sin(self.m), device=logits.device)  # sin(m)
        
        logits = logits * self.s  

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index, labels[index]] = False  # Mark target positions as False

        target_logit = logits[index, labels[index].view(-1)]
        non_target_logits = logits[mask].view(len(index), -1)
        
        diff_logits=non_target_logits-target_logit.unsqueeze(1)
        return diff_logits, sin_theta, sin_theta_plus_m, sin_m.expand(batch_size)

class CosFace_norm(torch.nn.Module):
    def __init__(self, s=64.0, m=0.35):
        super(CosFace_norm, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        batch_size = index.size(0)
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index, labels[index]] = False  # Mark target positions as False
        # Extract non-target logits
        non_target_logits = logits[mask].view(len(index), -1)

        with torch.no_grad():
            theta = target_logit.acos()                        # [batch_size]
            theta_plus_m=final_target_logit.acos()
            # 计算调整后的cos(θ + m)
            sin_theta = theta.sin()                            # sin(θ)
            sin_theta_plus_m = theta_plus_m.sin()          # sin(θ + m)
            sin_m =torch.tensor( math.sin(theta_plus_m-theta), device=logits.device)  # sin(m)

        diff_logits=non_target_logits-final_target_logit.unsqueeze(1)
        diff_logits = self.s*diff_logits  
        return diff_logits, sin_theta, sin_theta_plus_m, sin_m.expand(batch_size)

class SphereFace_norm(torch.nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, s=30., m=1.5):
        super(SphereFace_norm, self).__init__()
        # self.feat_dim = feat_dim
        # self.num_class = num_class
        self.s = s
        self.m = m
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        batch_size = index.size(0)
        target_logit = logits[index, labels[index].view(-1)]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index, labels[index]] = False  # Mark target positions as False
        # Extract non-target logits
        non_target_logits = logits[mask].view(len(index), -1)

        with torch.no_grad():
            theta = target_logit.acos()
            m_theta=self.m*theta
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            final_target_logit = sign * torch.cos(m_theta) - 2. * k

            theta_plus_m=final_target_logit.acos()
            # 计算调整后的cos(θ + m)
            sin_theta = theta.sin()                            # sin(θ)
            sin_theta_plus_m = theta_plus_m.sin()          # sin(θ + m)
            sin_m =torch.tensor( math.sin(theta_plus_m-theta), device=logits.device)  # sin(m)

        diff_logits=non_target_logits-final_target_logit.unsqueeze(1)
        diff_logits = self.s*diff_logits 
        # loss = F.cross_entropy(logits, y)

        return diff_logits, sin_theta, sin_theta_plus_m, sin_m.expand(batch_size)

class PowerFace_norm(torch.nn.Module):
    def __init__(self, s=64.0, m=0.7):
        super(PowerFace_norm,self).__init__()
        self.s=s
        self.m=m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        batch_size = index.size(0)

        with torch.no_grad():
            theta=target_logit.arccos()
            theta_plus_m=(theta/torch.pi).pow(self.m)*torch.pi

            final_target_logit = (theta_plus_m).cos()
            logits[index, labels[index].view(-1)] = final_target_logit

            sin_theta = theta.sin()                            # sin(θ)
            sin_theta_plus_m = theta_plus_m.sin()          # sin(θ + m)
            sin_m =(theta_plus_m-theta).sin()

        logits = logits * self.s 
        
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index, labels[index]] = False  # Mark target positions as False

        target_logit = logits[index, labels[index].view(-1)]
        non_target_logits = logits[mask].view(len(index), -1)
        
        diff_logits=non_target_logits-target_logit.unsqueeze(1)
        return diff_logits, sin_theta, sin_theta_plus_m, sin_m.expand(batch_size)

class PowerFace_norm1(torch.nn.Module):
    def __init__(self, s=64.0, m=0.7):
        super(PowerFace_norm1,self).__init__()
        self.s=s
        self.m=m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        batch_size = index.size(0)

        with torch.no_grad():
            theta=target_logit.arccos()
            theta_plus_m=(theta/torch.pi).pow(self.m)*torch.pi

            final_target_logit = (theta_plus_m).cos()
            logits[index, labels[index].view(-1)] = final_target_logit

            theta_m =theta_plus_m-theta

        logits = logits * self.s 
        
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[index, labels[index]] = False  # Mark target positions as False

        target_logit = logits[index, labels[index].view(-1)]
        non_target_logits = logits[mask].view(len(index), -1)
        
        diff_logits=non_target_logits-target_logit.unsqueeze(1)
        return diff_logits, theta_m

class NaiveFace(torch.nn.Module):
    def __init__(self,  s=64.0):
        super(NaiveFace, self).__init__()
        self.s = s

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits * self.s
        return logits
    
    

class PowerFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.6):
        super(PowerFace,self).__init__()
        self.s=s
        self.m=m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            final_target_logit = (target_logit/math.pi).pow(self.m)*math.pi
            final_target_logit.cos_()
            logits[index, labels[index].view(-1)] = final_target_logit

        logits = logits * self.s
        return logits
    
class PowerFace_d(torch.nn.Module):
    def __init__(self, s=64.0, m=10086,num_class=1000):
        super(PowerFace_d,self).__init__()
        self.s=s
        self.m=m
        self.num_class=num_class

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()


            d_m=torch.log(((average_n_cosine+math.log(self.num_class-1)/self.s).arccos()/math.pi))/torch.log((average_p_theta/math.pi))


        with torch.no_grad():
            target_logit.arccos_()
            final_target_logit = (target_logit/math.pi).pow(d_m)*math.pi
            final_target_logit.cos_()
            logits[index, labels[index].view(-1)] = final_target_logit

        logits = logits * self.s
        return logits

class PowerFace_dd(torch.nn.Module):
    def __init__(self, s=64.0, m=10086,num_class=1000):
        super(PowerFace_dd,self).__init__()
        self.s=s
        self.m=m
        self.num_class=num_class

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()


            d_m=torch.log(((average_n_cosine).arccos()/math.pi))/torch.log((average_p_theta/math.pi))


        with torch.no_grad():
            target_logit.arccos_()
            final_target_logit = (target_logit/math.pi).pow(d_m)*math.pi
            final_target_logit.cos_()
            logits[index, labels[index].view(-1)] = final_target_logit

        logits = logits * self.s
        return logits

class CosSinFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.6):
        super(CosSinFace,self).__init__()
        self.s=s
        self.m=m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        # target_logit = logits[index, labels[index].view(-1)]
        theta_logits=torch.arccos(logits)
        sin_logits=0.1*(torch.sin(2*theta_logits))
        logits= logits - sin_logits
        logits[index, labels[index].view(-1)] -= self.m
        logits = logits * self.s
        return logits

class ArcSinFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcSinFace, self).__init__()
        self.s = s
        self.margin = margin


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()

        theta_logits=torch.arccos(logits)
        sin_logits=0.1*(torch.sin(2*theta_logits))
        logits=logits-sin_logits

        logits = logits * self.s   
        return logits

class Forward_Back_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, s):
       ctx.save_for_backward(input)
       ctx.s=s
       output=input*s
       return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input,= ctx.saved_tensors
        return grad_output*ctx.s, None
    
class Forward_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, s):
       ctx.save_for_backward(input)
       ctx.s=s
       output=input*s
       return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input,= ctx.saved_tensors
        return grad_output, None

class Back_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, s):
       ctx.save_for_backward(input)
       ctx.s=s
       return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input,= ctx.saved_tensors
        return grad_output*ctx.s, None
    
class ArcFace_s(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5, f=Forward_Back_s):
        super(ArcFace_s, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False
        self.f=f


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()  
        logits=self.f.apply(logits, self.s)
        return logits
    
class TanFace_s(torch.nn.Module):
    def __init__(self, s=64.0, m1=0.6, m2=0.4, f=Forward_Back_s):
        super(TanFace_s,self).__init__()
        self.s=s
        self.m1=0.5
        self.m2=0.4
        self.f=f

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        target_logit.arccos_()
        final_target_logit = self.m1 * (-target_logit + math.pi/2)
        final_target_logit.tan_()
        logits[index, labels[index].view(-1)] = final_target_logit-self.m2
        logits = self.f.apply(logits,self.s)
        return logits
    

    
class ArcFace_forward_back_s(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace_forward_back_s, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()  
        logits=Forward_Back_s.apply(logits, self.s)
        return logits
    

    
class ArcFace_forward_s(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace_forward_s, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()  
        logits=Forward_s.apply(logits, self.s)
        return logits
    
class CosFace_d(torch.nn.Module):
    def __init__(self, s=64.0, m=10086, num_class=1000):
        super(CosFace_d, self).__init__()
        self.s = s
        self.m = m
        self.num_class=num_class

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()


            d_m=average_p_cosine-average_n_cosine-(math.log(self.num_class-1))/self.s

        final_target_logit = target_logit - d_m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

class CosFace_dd(torch.nn.Module):
    def __init__(self, s=64.0, m=10086, num_class=1000):
        super(CosFace_dd, self).__init__()
        self.s = s
        self.m = m
        self.num_class=num_class

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()


            d_m=average_p_cosine-average_n_cosine

        final_target_logit = target_logit - d_m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
    
class TanFace(torch.nn.Module):
    def __init__(self, s=64.0, m1=0.6, m2=0.4):
        super(TanFace,self).__init__()
        self.s=s
        self.m1=m1
        self.m2=m2

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        target_logit.arccos_()
        final_target_logit = self.m1 * (-target_logit + math.pi/2)
        final_target_logit.tan_()
        logits[index, labels[index].view(-1)] = final_target_logit-self.m2
        logits = logits * self.s
        return logits
    
class TanFace_a(torch.nn.Module):
    def __init__(self, s=64.0, m1=0.6, m21=0.4,m22=0.4):
        super(TanFace_a,self).__init__()
        self.s=s
        self.m1=m1
        self.m21=m21
        self.m22=m22

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        logits.arccos_()

        logits=self.m1 * (-logits + math.pi/2)
        logits.tan_()

        logits=logits-self.m21
        logits[index, labels[index].view(-1)]-=self.m22
        logits = logits * self.s
        return logits   
    
class SphereFace_d(torch.nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, s=32., m=10086,num_class=1000):
        super(SphereFace_d, self).__init__()
        # self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        # with torch.no_grad():
            # self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        # cos_theta = F.normalize(x, dim=1).mm(self.w)

        with torch.no_grad():
            index = torch.where(y != -1)[0]
            target_logit = x[index, y[index].view(-1)]
            postive_cosines=target_logit.clone()

            mask=torch.ones(x.size(),dtype=bool)
            mask[index,y[index].view(-1)]=False
            negative_cosines=x[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()

            d_m=(average_n_cosine+(math.log(self.num_class-1))/self.s).arccos()/average_p_theta


        with torch.no_grad():
            m_theta = torch.acos(x)
            m_theta.scatter_(
                1, y.view(-1, 1), d_m.item(), reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - x

        logits = self.s * (x + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits

class SphereFace_dd(torch.nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, s=32., m=10086,num_class=1000):
        super(SphereFace_dd, self).__init__()
        # self.feat_dim = feat_dim
        self.num_class = num_class
        self.s = s
        self.m = m
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

    def forward(self, x, y):
        # weight normalization
        # with torch.no_grad():
            # self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        # cos_theta = F.normalize(x, dim=1).mm(self.w)

        with torch.no_grad():
            index = torch.where(y != -1)[0]
            target_logit = x[index, y[index].view(-1)]
            postive_cosines=target_logit.clone()

            mask=torch.ones(x.size(),dtype=bool)
            mask[index,y[index].view(-1)]=False
            negative_cosines=x[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()

            d_m=(average_n_cosine).arccos()/average_p_theta


        with torch.no_grad():
            m_theta = torch.acos(x)
            m_theta.scatter_(
                1, y.view(-1, 1), d_m.item(), reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - x

        logits = self.s * (x + d_theta)
        # loss = F.cross_entropy(logits, y)

        return logits

class ArcFace_d(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, m=0.5,num_class=1000):
        super(ArcFace_d, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.num_class=num_class
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()

            d_m=(average_n_cosine+(math.log(self.num_class-1))/self.s).arccos()-average_p_theta

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + d_m
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits
    
class ArcFace_dd(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, m=0.5,num_class=1000):
        super(ArcFace_dd, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.num_class=num_class
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            postive_cosines=target_logit.clone()

            mask=torch.ones(logits.size(),dtype=bool)
            mask[index,labels[index].view(-1)]=False
            negative_cosines=logits[mask]

            average_p_cosine=postive_cosines.mean()
            average_n_cosine=negative_cosines.mean()

            average_p_theta=average_p_cosine.arccos()

            d_m=(average_n_cosine).arccos()-average_p_theta

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + d_m
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits

class SphereFaceRv2(torch.nn.Module):
    """ reference: <SphereFace: Deep Hypersphere Embedding for Face Recognition>"
        It also used characteristic gradient detachment tricks proposed in
        <SphereFace Revived: Unifying Hyperspherical Face Recognition>.
    """
    def __init__(self, s=60., m=1.4):
        super(SphereFaceRv2, self).__init__()
        # self.feat_dim = feat_dim
        # self.num_class = num_class
        self.s = s
        self.m = m
        # self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        # nn.init.xavier_normal_(self.w)

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        index = torch.where(y != -1)[0]    # 有效样本
        pos = y[index].view(-1)            # 正类索引

        with torch.no_grad():
            # 构造一个 mask，标记负类
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[index, pos] = False       # 正类位置 = False (不修改)

            # 对负类 logits 做变换
            neg_logits = x[mask].clone()
            theta = neg_logits.arccos()
            final_neg_logits = (theta/self.m).cos()

            # 写回负类位置
            x[mask] = final_neg_logits

        # scale
        x = x * self.s
        return x


