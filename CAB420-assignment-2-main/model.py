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

def getData2(direct_path, batch_size):
    # Path for images and GT
    imgPath = direct_path + "\\" + "train_img"
    gtPath = direct_path + "\\" + "train_gt"
    depthPath = direct_path + "\\" + "train_depth"
    # Load list of files in specified directory
    Imgfiles = os.listdir(imgPath)
    GTfiles = os.listdir(gtPath)
    Dfiles = os.listdir(depthPath)
    # Assign Lists to append to
    for index in range(0, number_of_items, batch_size):
        Images = []
        GT = []
        counts = []
        offset = index % 1000
        for i in range(batch_size):
            # Read Image and matlab GT files
            image = np.array(cv.imread(imgPath + "\\" + Imgfiles[offset + i]))
            curGT = np.array(sio.loadmat(
                gtPath + "\\" + GTfiles[offset + i])['point'])

            # depth = np.array(sio.loadmat(
            #     depthPath + "\\" + Dfiles[offset + i])['depth'])

            density = imDensity(image, curGT)
            # plt.imshow(image)
            #print(image.shape)
            depth[(depth < 0) | (depth > 20000)] = 20000
            depth /= 20000
            depth *= 255
            #depth = np.expand_dims(depth, axis=2).astype(int)
            depth = 255 - depth
            depth = cv.resize(depth, (resize_Width, resize_Height))
            depth = np.expand_dims(depth, axis=2).astype(int)

            image = cv.resize(image, (resize_Width, resize_Height))

            image = np.concatenate([image, depth], axis=2)

            # Append to existing lists
            Images.append(image)
            GT.append(cv.resize(density, (resize_Width, resize_Height)))
            counts.append(len(curGT))

        # Output with x and y datasets.
        yield np.array(Images), [np.array(GT), np.array(counts)]
        
        
def getData3(direct_path, batch_size, start_index, end_index):
    # Path for images and GT
    imgPath = direct_path + "\\" + "train_img"
    gtPath = direct_path + "\\" + "train_gt"
    depthPath = direct_path + "\\" + "train_depth"
    # Load list of files in specified directory
    Imgfiles = os.listdir(imgPath)
    GTfiles = os.listdir(gtPath)
    Dfiles = os.listdir(depthPath)
    # Assign Lists to append to
    # for index in range(0, number_of_items, batch_size):
    while (True):
        Images = []
        GT = []
        counts = []
        for i in range(batch_size):
            index = np.random.randint(start_index, end_index - batch_size)
            # Read Image and matlab GT files
            image = np.array(cv.imread(imgPath + "\\" + Imgfiles[index]))
            curGT = np.array(sio.loadmat(
                gtPath + "\\" + GTfiles[index])['point'])

            # depth = np.array(sio.loadmat(
            #     depthPath + "\\" + Dfiles[index])['depth'])

            density = imDensity(image, curGT)
            plt.imshow(image)
            #print(image.shape)
            # depth[(depth < 0) | (depth > 20000)] = 20000
            # depth /= 20000
            # depth *= 255
            # #depth = np.expand_dims(depth, axis=2).astype(int)
            # depth = 255 - depth
            # depth = cv.resize(depth, (resize_Width, resize_Height))
            # depth = np.expand_dims(depth, axis=2).astype(int)

            image = cv.resize(image, (resize_Width, resize_Height))

            # image = np.concatenate([image, depth], axis=2)

            # Append to existing lists
            Images.append(image)
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
loss_weights=[1, 0.001]
model_cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=['mse','mse'], loss_weights=loss_weights)
##model_cnn.fit()

#%%
## Model fit
model_fit = 1
epochs=10
batch_size=5
generator_number=5
steps_per_epoch=300

generator_train = getData3('./data/train_data', generator_number, 0, 1000)
generator_val = getData3('./data/train_data', generator_number, 1000, 1200)
generator_test = getData3('./data/test_data', generator_number, 0, 1000)
#generator = getData2('./data/train_data', generator_number)
#generator = getData3('./data/train_data', generator_number, 0, 1000)
#generator_val = getData3('./data/train_data', generator_number, 1000, 1200)


def modelfit():
    with tf.device('/gpu:0'):
        modelTrainHist = model_cnn.fit(generator_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=generator_val,
                            validation_steps=5)
        temptime = str(time.time())
        model_cnn.save('saved_model/'+ str(temptime) + "/my_model")
        my_file = open('saved_model/'+ str(temptime) + '/'+ str(temptime) + ".txt","w+")
        my_file.write('Loss Weights: ' + str(loss_weights[0]) + ', ' + str(loss_weights[1]) 
                      + '\nEpochs: ' + str(epochs) 
                      + '\nBatch Size: ' + str(batch_size) + 
                      '\nSteps Per Epoch: ' + str(steps_per_epoch))
def lasteModified():
    folder_name = ''
    last_modified = 0.00
    for dir in os.listdir('./saved_model'):
        if(dir != 'my_model'):
            if(float(dir) > float(last_modified)):
                last_modified = float(dir)

    return(str(last_modified))

if (model_fit == 1) :
    modelfit()

loadmodel = 'saved_model/' + "1622796765.2982879" + '/my_model'
print(loadmodel)
model_cnn = tf.keras.models.load_model(loadmodel)
#model_cnn.summary()
#%%
T_IMAGE, [T_DENSE, T_GROUND_TRUTH] = next(generator_train)
[T_DENSITY_PRED, T_PRED] = model_cnn.predict(T_IMAGE)
#%%
V_IMAGE, [V_DENSE, V_GROUND_TRUTH] = next(generator_val)
[V_DENSITY_PRED, V_PRED] = model_cnn.predict(V_IMAGE)

TE_IMAGE, [TE_DENSE, TE_GROUND_TRUTH] = next(generator_test)
[TE_DENSITY_PRED, TE_PRED] = model_cnn.predict(TE_IMAGE)
#%%
# print("Prediction: {0}".format(out2))
# plt.imshow(out1[0,:,:,0])

# print("Gnd shape:", GROUND_TRUTH.shape)
# print("Prediction shape:", GROUND_TRUTH.shape)

# Training
fig, ax = plt.subplots(2,1, figsize=[15,10])
ax[0].plot(T_GROUND_TRUTH, label="Ground Truth")
ax[0].plot(T_PRED, label="Prediction")
ax[0].legend()
t_results = tf.keras.losses.mean_squared_error(T_GROUND_TRUTH, T_PRED)
ax[0].set_title("Training \n Average MSE: {0:.0f}".format(np.average(t_results)))
# Validation
ax[1].plot(V_GROUND_TRUTH, label="Ground Truth")
ax[1].plot(V_PRED, label="Prediction")
ax[1].legend()
v_results = tf.keras.losses.mean_squared_error(V_GROUND_TRUTH, V_PRED)
ax[1].set_title("Validation \n Average MSE: {0:.0f}".format(np.average(v_results)))

# %%
# Testing
number_of_items_in_test_data = 100 #???

fig, ax = plt.subplots(figsize=[15,10])
ax.plot(TE_GROUND_TRUTH, label="Ground Truth")
ax.plot(TE_PRED, label="Prediction")
ax.legend()
te_results = tf.keras.losses.mean_squared_error(TE_GROUND_TRUTH, TE_PRED)
ax.set_title("Testing \n Average MSE: {0:.0f}".format(np.average(te_results)))
#%%
