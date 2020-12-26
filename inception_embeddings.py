
import numpy as np
from skimage.transform import resize
from keras.applications.inception_resnet_v2 import preprocess_input
from Inception import inception
def inception_embedding(gray_rgb):
    def resize_gray(x):
        return resize(x, (299, 299, 3), mode='constant')
    rgb = np.array([resize_gray(x) for x in gray_rgb])
    rgb = preprocess_input(rgb)
    embed = inception.predict(rgb)
    return embed