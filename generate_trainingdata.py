
from loading_train_data import train_X
from generate_data import datagen
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from inception_embeddings import inception_embedding
import json
with open('parameters.json') as f:
  data = json.load(f)
def gen_train(dataset=train_X, batch_size = data["batch_size"]):
    for batch in datagen.flow(dataset, batch_size=batch_size):
        X_batch = rgb2gray(batch)
        rgb = gray2rgb(X_batch)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield [X_batch, inception_embedding(rgb)], Y_batch