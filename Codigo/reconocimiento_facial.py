import pandas as pd
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('attr_celeba_prepared.txt', sep=' ',header=None)
df_train, df_test = train_test_split(df, test_size = 0.2)


files_train = tf.data.Dataset.from_tensor_slices(df_train[0])
attributes_train = tf.data.Dataset.from_tensor_slices(df_train.iloc[:,1:].to_numpy())
data_train = tf.data.Dataset.zip((files_train,attributes_train))

files_test = tf.data.Dataset.from_tensor_slices(df_test[0])
attributes_test = tf.data.Dataset.from_tensor_slices(df_test.iloc[:,1:].to_numpy())
data_test = tf.data.Dataset.zip((files_test,attributes_test))

path_to_images = '../img_align_celeba/'
def process_file(file_name,attributes):
    image = tf.io.read_file(path_to_images+file_name)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image, attributes


train_images = data_train.map(process_file)
test_images = data_test.map(process_file)

#for image, attri in train_images.take(2):
#     plt.imshow(image)
#     plt.show()
