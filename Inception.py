
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import json
with open('parameters.json') as f:
  data = json.load(f)

weights_path = data['Inception_weights']
try:
    inception = InceptionResNetV2(weights=weights_path, include_top=True)
except:
    inception = InceptionResNetV2(weights='Imagenet', include_top=True)