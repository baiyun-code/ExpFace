import os
import pickle
from PIL import Image
import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

path = "/home/baiyun/dev/data/faces_emore"
val_name_list = ["lfw", "cplfw", "calfw", "cfp_ff", "cfp_fp", "agedb_30", "vgg2_fp"]
image_size=(112, 112)

for val_name in val_name_list:
    print(f"Converting {val_name}")
    val_folder = os.path.join(path, val_name)
    os.makedirs(val_folder, exist_ok=True)
    imgs_folder = os.path.join(val_folder, "imgs")
    os.makedirs(imgs_folder, exist_ok=True)
    
    f = open(os.path.join(path, val_name+".bin"), 'rb')
    bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        
        i1 = Image.fromarray(img.asnumpy())
        i1.save(os.path.join(imgs_folder, f"{i}.jpg"))
        # i1.save(os.path.join(imgs_folder, f"{i}.png"))
        
        if i % 1000 == 0:
            print(f'Converted {i} images')

    np.save(os.path.join(val_folder, "pair_label"), issame_list)
