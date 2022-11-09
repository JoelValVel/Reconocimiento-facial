from dataclasses import replace
import pandas as pd
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import os


with open('list_attr_celeba.txt', 'r') as f:
    f.readline() #headers
    with open('attr_celeba_prepared1.txt', 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            new_line1 = new_line.replace("-1","0")
            newf.write(new_line1)
            newf.write('\n')


df = pd.read_csv('attr_celeba_prepared1.txt', sep=' ',header=None)

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files,attributes))

path_to_images = '../img_align_celeba/'
def process_file(file_name,attributes):
    image = tf.io.read_file(path_to_images+file_name)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)


for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()
