from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from autoencoder import autoencoder
from generate_trainingdata import gen_train
from loading_train_data import train_X


import json

with open('parameters.json',) as f:
  data = json.load(f)

filepath = data['model_path']
checkpoint = ModelCheckpoint(filepath,
                             save_best_only=True,
                             monitor='loss',
                             mode='min')

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            verbose=1, 
                                            factor=0.5,
                                            patience=3, 
                                            min_lr=0.00001)


model_callbacks = [learning_rate_reduction,checkpoint]
print('compiling model')


BATCH_SIZE = data["batch_size"]
model = autoencoder()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print('training started')

k = gen_train(train_X,BATCH_SIZE)
print(train_X.shape[0]/BATCH_SIZE)
model.fit_generator(gen_train(train_X,BATCH_SIZE),
            epochs=1,
            verbose=1,
            steps_per_epoch=train_X.shape[0]/BATCH_SIZE,
             callbacks=model_callbacks
                   )

