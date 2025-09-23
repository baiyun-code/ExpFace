import os
import mxnet as mx
import numpy as np

from PIL import Image

path_imgrec = os.path.join("../data/CASIA-WebFace", 'train.rec')
path_imgidx = os.path.join("../data/CASIA-WebFace", 'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)

if header.flag > 0:
    header0 = (int(header.label[0]), int(header.label[1]))
    imgidx = np.array(range(1, int(header.label[0])))
else:
    imgidx = np.array(list(imgrec.keys))

dir_idx = -1
img_idx = -1
l = len(imgidx)
print(l)
for j in range(0, l):
    s = imgrec.read_idx(imgidx[j])
    header, img = mx.recordio.unpack(s)
    
    if dir_idx != header.label[0]:
        dir_idx = int(header.label[0])
        print(f"Converting ID: {dir_idx}")
        os.makedirs(f"../data/CASIA-WebFace/imgs/{dir_idx}")

        img_idx = 1

    sample = mx.image.imdecode(img).asnumpy()
    i = Image.fromarray(sample)
    i.save(f"../data/CASIA-WebFace/imgs/{dir_idx}/{img_idx}.jpg", quality=95)

    img_idx += 1

