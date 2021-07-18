# Import Libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import time
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import cv2
import shutil
from sklearn.metrics import confusion_matrix, classification_report

# Create Dataframe from Images

tumor_dir='C:/Users/mozam/PycharmProjects/pythonProject/Brain Tumor Detecton _ Deep Learning/Brain MRI Images/yes'
healthy_dir='C:/Users/mozam/PycharmProjects/pythonProject/Brain Tumor Detecton _ Deep Learning/Brain MRI Images/no'
dirlist=[tumor_dir, healthy_dir]
classes=['Brain Tumor', 'Healthy']
filepaths=[]
labels=[]
for d,c in zip(dirlist, classes):
    flist=os.listdir(d)
    for f in flist:
        fpath=os.path.join (d,f)
        filepaths.append(fpath)
        labels.append(c)
#print ('filepaths: ', len(filepaths), '   labels: ', len(labels))

Fseries=pd.Series(filepaths, name='file_paths')
Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries,Lseries], axis=1)
df=pd.DataFrame(np.array(df).reshape(253,2), columns = ['file_paths', 'labels'])
#print(df['labels'].value_counts())

# Visualize MRI Images

plt.figure(figsize=(14, 10))
for i in range(10):
    random = np.random.randint(1, len(df))
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.imread(df.loc[random, "file_paths"]))
    plt.title(df.loc[random, "labels"], size=10, color="black")
    plt.xticks([])
    plt.yticks([])

plt.show()

# Train, Valid, Test Dataframe Splits

train_df, test_df = train_test_split(df, train_size=0.95, random_state=0)
train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)

#print(train_df.labels.value_counts())
#print(valid_df.labels.value_counts())
#print(test_df.labels.value_counts())

# Image Data Generator

target_size=(299,299)
batch_size=64

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input, zoom_range=0.1, horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
train_gen = train_datagen.flow_from_dataframe(train_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(valid_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(test_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')