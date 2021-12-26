#%%
# Imports 
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorboard import notebook
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.layers import PReLU
from tensorflow import keras as keras
from tensorflow.keras.layers import concatenate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pydot
import IPython
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot, plot_model

## Imports
import pandas as pd
import cv2 as cv
import scipy.io as sio
import math
import time

#%%
tf.test.is_built_with_cuda()
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#%%
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def imDensity(im, points):
    h = 1080
    w = 1920
    im_density = np.zeros((h, w))

    if(len(points) == 0):
        return

    if(len(points[:, 0]) == 1):
        x1 = max(1, min(w, round(points[0, 0])))
        y1 = max(1, min(h, round(points[0, 1])))
        im_density[y1, x1] = 255
        return

    for j in range(1, len(points)):
        f_sz = 15
        sigma = 4.0
        H = matlab_style_gauss2D([f_sz, f_sz], sigma)
        x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(math.floor(points[j, 1])))))

        if(x > w or y > h):
            continue

        x1 = x - int(math.floor(f_sz/2))
        y1 = y - int(math.floor(f_sz/2))
        x2 = x + int(math.floor(f_sz/2))+1
        y2 = y + int(math.floor(f_sz/2))+1
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        if(x1 < 1):
            dfx1 = abs(x1)+1
            x1 = 1
            change_H = True

        if(y1 < 1):
            dfy1 = abs(y1)+1
            y1 = 1
            change_H = True

        if(x2 > w):
            dfx2 = x2 - w
            x2 = w
            change_H = True

        if(y2 > h):
            dfy2 = y2 - h
            y2 = h
            change_H = True

        x1h = 1+dfx1
        y1h = 1+dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if (change_H == True):
            H = matlab_style_gauss2D(
                [float(y2h-y1h+1), float(x2h-x1h+1)], sigma)
        im_density[y1: y2, x1: x2] = im_density[y1: y2, x1: x2] + H
    return im_density

number_of_items = 100000000000000
resize_Width = int(1920 / 4)
resize_Height = int(1080 / 4)
def getData(direct_path, batch_size):
    # Path for images and GT
    imgPath = direct_path + "\\" + "train_img"
    gtPath = direct_path + "\\" + "train_gt"
    # Load list of files in specified directory
    Imgfiles=os.listdir(imgPath)
    GTfiles=os.listdir(gtPath)
    # Assign Lists to append to
    for index in range(0,number_of_items,batch_size):
        Images = []
        GT = []
        counts = []
        offset = index % 1000
        for i in range(batch_size):
            # Read Image and matlab GT files
            image  = np.array(cv.imread(imgPath + "\\" + Imgfiles[offset + i]))
            curGT  = np.array(sio.loadmat(gtPath + "\\" + GTfiles[offset + i])['point'])

            # Turn points into density map
            density = imDensity(image, curGT)

            # Append to existing lists
            Images.append(cv.resize(image, (resize_Width, resize_Height))) 
            GT.append(cv.resize(density, (resize_Width, resize_Height)))
            counts.append(len(curGT))
        
        # Output with x and y datasets.
        yield np.array(Images), [np.array(GT), np.array(counts)]

#%%
with tf.device('/cpu:0'):
    # Model
    inputs = keras.Input(shape=(resize_Height, resize_Width, 3, ), name='img')
    ## Base Layer
    x = layers.Conv2D(filters=16, kernel_size=(9,9), padding="same", activation=PReLU())(inputs)
    x = layers.Conv2D(filters=32, kernel_size=(7,7), padding="same", activation=PReLU())(x)

    ## High-level Priority Stage
    HP = layers.Conv2D(filters=16,kernel_size=(9,9), padding="same", activation=PReLU())(x)
    HP = layers.MaxPool2D(pool_size=(2, 2))(HP)
    HP = layers.Conv2D(filters=32,kernel_size=(7,7), padding="same", activation=PReLU())(HP)
    HP = layers.MaxPool2D(pool_size=(2, 2))(HP)
    HP = layers.Conv2D(filters=16,kernel_size=(7,7), padding="same", activation=PReLU())(HP)
    HP = layers.Conv2D(filters=8,kernel_size=(7,7), padding="same", activation=PReLU())(HP)

    ## Classification Output
    CO = layers.GlobalMaxPool2D()(HP)
    CO = layers.Flatten()(CO)
    CO = layers.Dense(512, activation='relu')(CO)
    CO = layers.Dense(256, activation='relu')(CO)
    OutputClass = layers.Dense(1, activation='relu')(CO)

    ## Density Estimator
    ## Stage 1

    DE = layers.Conv2D (filters=20, kernel_size=(7,7), padding="same", activation=PReLU())(x)
    DE = layers.MaxPool2D(pool_size=(2, 2))(DE)
    DE = layers.Conv2D (filters=40, kernel_size=(5,5), padding="same", activation=PReLU())(DE)
    DE = layers.MaxPool2D(pool_size=(2, 2))(DE)
    DE = layers.Conv2D (filters=20, kernel_size=(5,5), padding="same", activation=PReLU())(DE)
    DE = layers.Conv2D (filters=10, kernel_size=(5,5), padding="same", activation=PReLU())(DE)

    ## Stage 2
    DE_HP = concatenate([DE, HP])
    DE = layers.Conv2D(filters=24, kernel_size=(3,3), padding="same", activation=PReLU())(DE_HP)
    DE = layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=PReLU())(DE)

    DE = layers.Conv2DTranspose(filters=16, kernel_size=(4,4), padding="same", use_bias=True)(DE)
    DE = layers.PReLU()(DE)
    DE = layers.Conv2DTranspose(filters=8, kernel_size=(4,4), padding="same", use_bias=True)(DE)
    DE = layers.PReLU()(DE)

    DE = layers.UpSampling2D (size=(4, 4))(DE)
    DE = layers.ZeroPadding2D(padding=(1,0))(DE)
    DE = layers.Conv2D (filters=1, kernel_size=(1,1), padding="same", activation='relu')(DE)


    # note both outputs defined here
    model_cnn = keras.Model(inputs=inputs, outputs=[DE, OutputClass], name='Assign2Model')
    model_cnn.summary()
#%%
#Show model
plot_model(model_cnn, to_file='Assignment2_v2.png', show_shapes=True)
IPython.display.Image('Assignment2_v2.png')
#%%
def density_loss2(y_true, y_pred):
    # Data is passed in by the batch
    batch_loss = []
    # Each item in batch
    for i in range(2):
        sub = y_true[i] - y_pred[i]
        loss = 0
        # Each pixel
        for y in range(540):
            for x in range(960):
                loss += abs(sub[y,x])
        batch_loss[i] = loss
    return batch_loss

#%%
def l2_loss(y_true, y_pred):

    y_true = K.repeat_elements(y_true, rep=5, axis=-1)

    loss = K.mean(K.square(y_true - y_pred))

    return loss
#%%
# Custom loss example for keras: https://keras.io/api/losses/ (scroll down to custom loss)
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)
model_cnn.compile(optimizer='adam', loss=['mse','mse'], loss_weights=[50,25])
##model_cnn.fit()

#%%
## Model fit
#generator = getData('./data/train_data', 45)
model_fit = 0
def modelfit():
    with tf.device('/gpu:0'):
        modelTrainHist = model_cnn.fit(generator,
                            epochs=10,
                            batch_size=45,
                            steps_per_epoch=3000)
        model_cnn.save('saved_model/'+ str(time.time()) + "/my_model")
        
def lasteModified():
    folder_name = ''
    last_modified = 0.00
    for dir in os.listdir('./saved_model'):
        if(dir != 'my_model'):
            if(float(dir) > float(last_modified)):
                last_modified = dir

    return(str(last_modified))

if (model_fit == 1) :
    modelfit()

model_cnn = tf.keras.models.load_model('saved_model/' + lasteModified() + '/my_model')
model_cnn.summary()

folder_name = ''
last_modified = ''
for dir in os.listdir('./saved_model'):
    if(dir != 'my_model'):
        if(float(dir) <= modified_time):
            folder_name = dir

print(folder_name)

#%%
IMAGE, [DENSE, GROUND_TRUTH] = next(generator)
#%%
print(GROUND_TRUTH)
[out1, out2] = model_cnn.predict(IMAGE)
print("Prediction: {0}".format(out2[0]))
plt.imshow(out1[0,:,:,0])
#%%


#%%
