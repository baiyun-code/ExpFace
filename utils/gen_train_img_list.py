from torchvision.datasets import ImageFolder
import pickle

dataset = ImageFolder("../../data/ms1mv3/imgs")
# dataset = ImageFolder("../Data/MS1MV3A/imgs")

a = open("../../data/ms1mv3.pickle", "wb")
# pickle.dump(dataset.imgs, a)
# a.close()

# a = open("../Data/casia-a.pickle", "wb")
pickle.dump(dataset.imgs, a)
a.close()

# b = open("../../Data/webface21m.pickle", "rb")
# c = pickle.load(b)
# b.close()