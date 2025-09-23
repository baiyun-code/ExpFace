from torchvision.datasets import ImageFolder
import pickle

addr_old = "C:/Users/Edward/Downloads/WebFace12M.pickle"
addr_new = "C:/Users/Edward/Downloads/WebFace12M_new.pickle"

root_old = "WebFace12M"
root_new = "WebFace42M"

file_old = open(addr_old, "rb")
imgs = pickle.load(file_old)
file_old.close()

for i in range(len(imgs)):
    imgs[i] = list(imgs[i])
    imgs[i][0] = imgs[i][0].replace(root_old, root_new)
    imgs[i] = tuple(imgs[i])
    
file_new = open(addr_new, "wb")
pickle.dump(imgs, file_new)
file_new.close()