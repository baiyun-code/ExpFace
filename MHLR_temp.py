import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("F:/Face_Recognition_Backup/wf42m_r100_pfc0.2_tau0.05_lambd5e-5/loss.csv")
c = a["lr"].values
y = a["loss"].values
n = y.shape[0]
x = np.arange(n)






alpha = 0.001
beta = 0.0001
iter_per_epoch = len(y) / 5
delay = len(y)*0.04
iter_epoch = iter_per_epoch
toler = delay
power = 0
lr = []
d = [alpha*y[0] - alpha*y[1]]
dema = [d[0]]
for i in range(1, a.shape[0]):
    if i%1000==0:
        print(i)
        
    d.append((1-alpha)*d[i-1]+alpha*(y[i-1]-y[i]))
    dema.append(beta*d[i]+(1-beta)*dema[i-1])
    
    lr.append(0)
    if i > iter_epoch:
        toler = max(toler, iter_epoch + delay)
        iter_epoch += iter_per_epoch
    
    if i > toler:
        if dema[i] < 0:
            # power += 1
            lr[-1] = 0.0003
            toler = i + delay
                
            # if power > 9:
                # lr[-1] = 0

# plt.figure()
# plt.plot(c)
plt.figure()
plt.plot(y[:])
# plt.figure()
# plt.plot(d[:])
plt.figure()
plt.plot(dema[:])
plt.plot(lr)
plt.hlines([0], [0], [toler])






# alpha = 0.001
# beta = 0.0001
# toler = len(y)*0.05
# thres = 0.00001
# count = 0
# power = 0
# lr = []
# d = [alpha*y[0] - alpha*y[1]]
# dema = [d[0]]
# for i in range(1, a.shape[0]):
#     if i%1000==0:
#         print(i)
        
#     d.append((1-alpha)*d[i-1]+alpha*(y[i-1]-y[i]))
#     dema.append(beta*d[i]+(1-beta)*dema[i-1])
    
#     lr.append(0)
#     if dema[i] < thres:
#         if count < toler:
#             count += 1
#         else:
#             count = 0
#             power += 1
#             lr[-1] = 0.001
            
#         if power > 9:
#             lr[-1] = 0
#     else:
#         count = 0
#         w_count = 0

# # plt.figure()
# # plt.plot(c)
# plt.figure()
# plt.plot(y[:])
# # plt.figure()
# # plt.plot(d[:])
# plt.figure()
# plt.plot(dema[:])
# plt.plot(lr)
# plt.hlines([thres], [0], [len(dema)])
# plt.hlines([0], [0], [toler])







# m = 4000
# k = np.ones(m)
# avg = np.convolve(y, k, mode="same") / m
# # k[:int(m/2)] = -1
# k1 = np.array([-1,1])
# diff = np.convolve(avg, k1, mode="same")
# y = diff[m+1:]
# ema = [y[0]]
# alpha = 0.999
# for i in range(1, y.shape[0]):
#     ema.append(alpha*ema[i-1]+(1-alpha)*y[i])

# plt.figure()
# plt.plot(avg[m+1:])

# plt.figure()
# plt.plot(diff[m+1:])

# plt.figure()
# plt.plot(ema)
