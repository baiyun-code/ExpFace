import torch
import time

# torch.manual_seed(128)
num_local = 2000000
num_sample = int(0.3*num_local)
weight = torch.randint(high=num_local, size=[num_local, 512], dtype=torch.float32)
labels = torch.randint(0, num_local, size=[128], dtype=torch.int32)






st = time.time()
a = labels.clone()
with torch.no_grad():
    positive = torch.unique(labels, sorted=True)
    print("positive----------------------------")
    print(positive)
    if num_sample - positive.size(0) >= 0:
        perm = torch.rand(size=[num_local])
        print("perm----------------------------")
        print(perm)
        perm[positive] = 2.0
        print("perm----------------------------")
        print(perm)
        index = torch.topk(perm, k=num_sample)[1]
        print("index----------------------------")
        print(index)
        index = index.sort()[0]
        print("index----------------------------")
        print(index)
        print(labels)
        print("----------------------------")
    else:
        index = positive
    weight_index = index

    a = torch.searchsorted(index, labels)
    print("----------------------------")
    print(index)
    print(a)
    print(labels)

b = weight[weight_index]
print("----------------------------")
print(b)
print(time.time() - st)








# st = time.time()
# l = torch.arange(num_local)
# with torch.no_grad():
#     positive, idx_ivs = torch.unique(labels, sorted=False, return_inverse=True)
#     print("positive----------------------------")
#     print(positive)
#     print("idx_ivs----------------------------")
#     print(idx_ivs)
#     index = torch.zeros(num_sample, dtype=torch.int32)
    
#     # negative = torch.zeros(num_local - positive.shape[0], dtype=torch.int32)
#     mask = torch.isin(l, positive, assume_unique=True, invert=True)
#     negative = l[mask]
#     print("negative----------------------------")
#     print(negative)
    
#     perm = torch.randperm(negative.shape[0])
#     print("perm----------------------------")
#     print(perm)
#     index[:positive.shape[0]] = positive
#     index[positive.shape[0]:] = negative[perm[:(num_sample - positive.shape[0])]]
#     print("index----------------------------")
#     print(index)
    
# b = weight[index]
# print("b----------------------------")
# print(b)
# print(time.time() - st)






# l = torch.arange(num_local)
# mask = torch.ones(num_local, dtype=torch.bool)

# st = time.time()
# with torch.no_grad():
#     print("labels----------------------------")
#     print(labels)
#     positive, idx_ivs = torch.unique(labels, sorted=False, return_inverse=True)
#     print("positive----------------------------")
#     print(positive)
#     print("idx_ivs----------------------------")
#     print(idx_ivs)
#     index = torch.zeros(num_sample, dtype=torch.int32)
    
#     mask[positive] = False
#     negative = l[mask]
#     mask[positive] = True
#     print("negative----------------------------")
#     print(negative)
    
#     perm = torch.randperm(negative.shape[0])
#     print("perm----------------------------")
#     print(perm)
#     index[:positive.shape[0]] = positive
#     index[positive.shape[0]:] = negative[perm[:(num_sample - positive.shape[0])]]
#     print("index----------------------------")
#     print(index)

# a = idx_ivs
# print("a----------------------------It is a question if this is needed")
# print(a)
# b = weight[index]
# print("b----------------------------")
# print(b)
# print(time.time() - st)