
from tensorflow import keras
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
import numpy as np
from inception_embeddings import inception_embedding

import json

with open('parameters.json') as f:
  data = json.load(f)

filepath = data['model_path']

model = keras.models.load_model(filepath)

TestImagePath=data['colo_images']
test = []

for file in tqdm(os.listdir(TestImagePath)):
    try:
        img = cv2.imread(TestImagePath+file)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256,256))
        test.append(img)
    except:
        pass
test = np.array(test).astype('float32') / 255.

im = gray2rgb(rgb2gray(test))
im_embed = inception_embedding(im)
im = rgb2lab(im)[:,:,:,0]
im = im.reshape(im.shape+(1,))

pred = model.predict([im, im_embed])
pred = pred * 128

decodings = np.zeros((len(pred),256, 256, 3))

for i in range(len(pred)):
    pp = np.zeros((256, 256, 3))
    pp[:,:,0] = im[i][:,:,0]
    pp[:,:,1:] = pred[i]
    decodings[i] = lab2rgb(pp)
    cv2.imwrite("img_"+str(i)+".jpg", lab2rgb(pp))

# recolored
plt.figure(figsize=(40, 10))
for i in range(5):
    plt.subplot(3, 10, i + 1 +10)
    plt.imshow(decodings[i].reshape(256, 256,3))
    plt.axis('off')
     
plt.tight_layout()
plt.show()