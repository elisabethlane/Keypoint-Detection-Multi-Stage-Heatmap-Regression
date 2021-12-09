import tensorflow as tf
from tensorflow import keras

from data import DataSet
from model import Model
from data_generator import DataGenerator

import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# constants
NUM_KEYPOINTS = 2
IMG_WIDTH = 192
IMG_HEIGHT = 192
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)
SIGMA = 5
BATCH_SIZE = 64

# model checkpoint directory
checkpoints_dir = "data/checkpoints/"
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

# get the data
data = DataSet(IMG_HEIGHT, IMG_WIDTH)
X_train = data.train_image_data
y_train = data.train_label_data
X_val = data.test_image_data
y_val = data.test_label_data

# get the model
model = Model(INPUT_SHAPE, NUM_KEYPOINTS)

# create data generators
gen_train = DataGenerator(X_train, y_train, SIGMA, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_KEYPOINTS)
gen_val = DataGenerator(X_val, y_val, SIGMA, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_KEYPOINTS)


# save the model
checkpoint_filepath = os.path.join('data', 'checkpoints', '{epoch:03d}-{val_loss:.10f}.hdf5')
checkpointer = ModelCheckpoint(filepath = checkpoint_filepath,
                                   monitor='val_loss', 
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False, 
                                   mode='auto')

early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta= 0,
                              patience= 30, 
                              restore_best_weights=True)

# train the model - normal
history = model.model.fit_generator(generator=gen_train, 
                                    epochs=1000, 
                                   validation_data=gen_val,                                  
                                   verbose=1,                                    
                                   callbacks=[early_stopper, 
                                   checkpointer])


# print(history.history.keys())

# loss_val = history.history['val_loss']
# loss_train = history.history['loss']

# epochs = range(1,len(loss_val)+1)
# plt.plot(epochs, loss_train, 'g', label='Training loss')
# plt.plot(epochs, loss_val, 'b', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

#tf.keras.utils.plot_model(model.model, to_file='model_img.png', show_shapes=True)
