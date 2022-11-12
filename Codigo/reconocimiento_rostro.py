from numpy import imag
import pandas as pd
import tensorflow as tf
import datetime
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD 
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
from PIL import Image
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainhomer = r'C:\\Users\\erjo_\\Downloads\\simpsons_dataset\\homer_simpson\\'
trainrandom = r'C:\\Users\\erjo_\\Downloads\\simpsons_dataset\\random\\'
homer_files_path = os.path.join(trainhomer, '*')
random_files_path = os.path.join(trainrandom, '*')
homer_files = sorted(glob(homer_files_path))
random_files = sorted(glob(random_files_path))
n_files = len(homer_files) + len(random_files)

size_image = 192
allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0

for f in homer_files:
    try:
        #img = io.imread(f)
        #new_img = imresize(img, (size_image, size_image, 3))
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
        #continue
for f in random_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue
x, x_test, y, y_test = train_test_split(allX, ally, test_size=0.2, random_state=1)


model = keras.models.load_model('modelo_final.h5')
print(model.summary())

for l in model.layers:
    l.trainable=False

model.layers[9].trainable = True

###

learning_rate = 0.00001
opt= SGD(learning_rate=learning_rate, momentum=0.9)
#opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon )
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)
#history = model.fit(train_images,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images), callbacks=[WandbCallback()])
log_dir="logs_simpson/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

batch_size = 100
epochs = 20

epoch_steps = len(y)//batch_size
test_steps = len(y_test)//batch_size

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

model.fit_generator(
                aug.flow(x,y,batch_size=batch_size),
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=(x_test,y_test),
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )
#score = model.evaluate(test_images, verbose=0)
#print(score)
model.save('modelo_simpson')

