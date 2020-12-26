import os
import cv2
import numpy as np
from tqdm import tqdm
import json
with open(r'C:\Users\amitn\Downloads\Deep_Learning-master\Deep_Learning-master\Image Colorization\parameters.json') as f:
  data = json.load(f)
path1 = "../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/drawings/"
path2 = "../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/engraving/"
path3 = "../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/iconography/"
path4 = data['dataset']
path5 = "../input/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/sculpture/"
image_size = 256
path = [path4] # you can add more path if you have higher RAM

train_X = []
for p in tqdm(path):
    print(p)
    for f in tqdm(os.listdir(p)):
        try:
            img = cv2.imread(p+'/'+f)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256,256))
            train_X.append(img)
        except:
            pass
train_X = np.array(train_X).astype('float32') / 255.
print(train_X)