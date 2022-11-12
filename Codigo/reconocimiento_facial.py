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
from tensorflow.keras.optimizers import SGD 
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback

df = pd.read_csv('attr_celeba_prepared1.txt', sep=' ',header=None)
df_red = df.sample(frac=0.5)
df_train, df_test = train_test_split(df_red, test_size = 0.2)


files_train = tf.data.Dataset.from_tensor_slices(df_train[0])
attributes_train = tf.data.Dataset.from_tensor_slices(df_train.iloc[:,2].to_numpy())
data_train = tf.data.Dataset.zip((files_train,attributes_train))

files_test = tf.data.Dataset.from_tensor_slices(df_test[0])
attributes_test = tf.data.Dataset.from_tensor_slices(df_test.iloc[:,2].to_numpy())
data_test = tf.data.Dataset.zip((files_test,attributes_test))

path_to_images = r'C:\\Users\\erjo_\\Downloads\\img_align_celeba\\'
def process_file(file_name,attributes):
    image = tf.io.read_file(path_to_images+file_name)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    image = tf.expand_dims(image,axis=0)
    attributes = tf.reshape(attributes,[1,1])
    return [image, attributes]


train_images = data_train.map(process_file)
test_images = data_test.map(process_file)

#for image, attri in train_images.take(2):
#    plt.imshow(image)
#     plt.show()

#print(test_images)

learning_rate = 0.00001
rho = 0.9
epsilon = 1e-7
batch_size = 100
epochs = 20

epoch_steps = len(train_images)//batch_size
test_steps = len(test_images)//batch_size

#wandb.init(project="reconocimiento_facial")
#wandb.config.learning_rate = learning_rate
#wandb.config.rho= rho
#wandb.config.epsilon = epsilon
#wandb.config.batch_size = batch_size
#wandb.config.epochs = epochs
#wandb.config.capas = """3 atributos,
#                        SGD,
#                        loss categorical_crossentropy"""

model = Sequential()
layer1 = Conv2D(10, (3, 3), input_shape=(192, 192,3),activation='relu') #
model.add(layer1)
layer2 = MaxPooling2D(pool_size=(2, 2))
model.add(layer2)
layer3 = Conv2D(10, (3, 3),activation='relu') #
model.add(layer3)
layer4 =MaxPooling2D(pool_size=(2, 2))
model.add(layer4)
layer5 = Conv2D(20, (3, 3), activation='relu') #
model.add(layer5)
layer6 = MaxPooling2D(pool_size=(2, 2))
model.add(layer6)
layer7 = Flatten()
model.add(layer7)
layer8 = Dense(64,activation='relu') #
model.add(layer8)
layer9 = Dropout(0.2)
model.add(layer9)
layer10 = Dense(1,activation='relu') #
model.add(layer10)
opt= SGD(learning_rate=learning_rate, momentum=0.9)
#opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=rho, epsilon=epsilon )
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)
#history = model.fit(train_images,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images), callbacks=[WandbCallback()])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

model.fit_generator(
                train_images,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test_images,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )
#score = model.evaluate(test_images, verbose=0)
#print(score)
model.save('modelo_inicial.h5')

for i in range(1,10):
    exec(f"layer{i}.trainable=False")


model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)

model.fit_generator(
                train_images,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test_images,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )
model.save('modelo_2.h5')

for i in range(1,10):
    exec(f"layer{i}.trainable=True")

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)

model.fit_generator(
                train_images,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test_images,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )
model.save('modelo_final.h5')