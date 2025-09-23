from torchvision.datasets import ImageFolder
import pickle

dataset = ImageFolder("../../Data/WebFace260M")

root_old = dataset.root
wf42m = dataset.imgs

# WebFace12M
root_new = "../Data/WebFace260M"
i_zip = len(root_old) + 1

i_end = 0
for img, _ in wf42m:
    if int(img[i_zip]) < 3:
        i_end += 1
    else:
        break

imgs = wf42m[:i_end]
for i in range(len(imgs)):
    imgs[i] = list(imgs[i])
    imgs[i][0] = imgs[i][0].replace(root_old, root_new)
    imgs[i] = tuple(imgs[i])
    
print(len(imgs))
print(imgs[-1][1]+1)

file = open("../../Data/webface12m.pickle", "wb")
pickle.dump(imgs, file)
file.close()

# b = open("../../Data/webface21m.pickle", "rb")
# c = pickle.load(b)
# b.close()
