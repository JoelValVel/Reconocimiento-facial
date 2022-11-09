from numpy import imag
import pandas as pd
import tensorflow as tf
import datetime
import pathlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop 
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback

df = pd.read_csv('attr_celeba_prepared1.txt', sep=' ',header=None)
df_red = df.sample(frac=0.05)
df_train, df_test = train_test_split(df_red, test_size = 0.2)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

files_train = tf.data.Dataset.from_tensor_slices(df_train[0])
#attributes_train = df_train.iloc[:,1:]
y_train = tf.convert_to_tensor([df_train.iloc[:,1:].iloc[i,:].tolist() for i in range(len(df_train))])
#data_train = tf.data.Dataset.zip((files_train,attributes_train))

files_test = tf.data.Dataset.from_tensor_slices(df_test[0])
y_test = tf.convert_to_tensor([df_test.iloc[:,1:].iloc[i,:].tolist() for i in range(len(df_test))])
#attributes_test = tf.data.Dataset.from_tensor_slices(df_test.iloc[:,1:].to_numpy())
#data_test = tf.data.Dataset.zip((files_test,attributes_test))

path_to_images = r'C:\\Users\\erjo_\Downloads\\img_align_celeba\\'
def process_file(file_name):
    image = tf.io.read_file(path_to_images+file_name)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    image = tf.expand_dims(image,axis=0)
    return tf.reshape(image,[192,192,3])


train_images = tf.convert_to_tensor([process_file(i) for i in files_train])
test_images = tf.convert_to_tensor([process_file(i) for i in files_test])
#train_images = tf.reshape(train_images,[len(),190,190,10])
#test_images = tf.reshape(test_images,[1,190,190,10])

#for image, attri in train_images.take(2):
#    plt.imshow(image)
#     plt.show()

#print(test_images)

learning_rate = 0.01
rho = 0.9
epsilon = 1e-7
batch_size = 100
epochs = 100

epoch_steps = len(train_images)//batch_size
test_steps = len(test_images)//batch_size

wandb.init(project="reconocimiento_facial")
wandb.config.learning_rate = learning_rate
wandb.config.rho= rho
wandb.config.epsilon = epsilon
wandb.config.batch_size = batch_size
wandb.config.epochs = epochs
wandb.config.capas = """Capa Conv2D(10,(3,3)) relu pool_size=(2,2), 
                        Capa Conv2D(10,(3,3)) relu pool_size(2,2),
                        Capa Conv2D(10,(3,3)) relu pool_size(2,2) flatten,
                        Capa densa 64 relu dropout 0.2,
                        Capa densa 40  sigmoid,
                        RMSProp,
                        loss binary_crossentropy"""

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(192, 192,3))) #
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, (3, 3))) #
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3))) #
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64)) #
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(40)) #
model.add(Activation('sigmoid'))
opt= tf.keras.optimizers.Adam(learning_rate=learning_rate)
#opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon )
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)
#history = model.fit(train_images,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images), callbacks=[WandbCallback()])
#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

#model.fit_generator(
#                train_images,
#                steps_per_epoch=epoch_steps,
#                epochs=epochs,
#                validation_data=test_images,
#                validation_steps=test_steps,
#                callbacks=[tbCallBack]
#                )
#score = model.evaluate(test_images, verbose=0)
#print(score)
history = model.fit(x=train_images, y=y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images,y_test), callbacks=[WandbCallback()])
score = model.evaluate(x=test_images, y=y_test, verbose=0)
print(score)