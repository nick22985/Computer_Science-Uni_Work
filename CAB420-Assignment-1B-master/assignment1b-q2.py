# %%
from pathlib import Path
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import matplotlib.pyplot as plt
import scipy.io
import numpy
from sklearn import decomposition
from sklearn import discriminant_analysis
import re
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import os
import matplotlib.pyplot as plt
import random
from tensorflow.python.client import device_lib 
from scipy.spatial import distance
import sys
import numpy as np
from datetime import datetime
#%%
tf.test.is_built_with_cuda()
#%%
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# %%
def readImage(path, name):
    folder = Path(path).rglob('*.jpg')
    files = [x for x in folder]
    ImageArray = []
    yarray = []
    for x in files:
        img = cv2.imread(str(x))
        # img = img.flatten('C')
        ImageArray.append(img)
        test = re.search("[0-9]{4}", str(x))
        yarray.append(int(test[0]))
    obj_arr = numpy.zeros((2,), dtype=numpy.object)
    obj_arr[0] = ImageArray
    obj_arr[1] = numpy.transpose([yarray])
    scipy.io.savemat('./Data/Q2/matfiles' + name + '.mat',
                     mdict={'x': obj_arr[0], 'y': obj_arr[1]})

def createFig(imgArray):
    # print(imgArray)
    fig = plt.figure(figsize=(50, 50))
    for i in range(10):
        ax = fig.add_subplot(50, 50, i+1)
        ax.imshow(imgArray[i, :, :, :])
        # ax.imshow(imgArray[:,:,:,i])
        
def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance');
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    conf.ax_.set_title('Test Set Performance');
    pred = model.predict(X_test)
    print('Test Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))

# %% Create 4D arrays for iamge vectors and for Y values. Then making mat files for the data to be stored
readImage('./Data/Q2/Testing/Gallery', 'Gallery')
readImage('./Data/Q2/Testing/Probe', 'Probe')
readImage('./Data/Q2/training', 'Training')

Training = scipy.io.loadmat('./Data/Q2/matfilesTraining.mat')
Probe = scipy.io.loadmat('./Data/Q2/matfilesProbe.mat')
Gallery = scipy.io.loadmat('./Data/Q2/matfilesGallery.mat')


#%%
training_x_Org = Training['x'].astype("float32") / 255.0
training_x = training_x_Org
training_y_Org = Training['y'].astype("float32")
training_y = training_y_Org
testing_Probe_x_Org = Probe['x'].astype("float32") / 255.0
testing_Probe_x = testing_Probe_x_Org
testing_Probe_y_Org = Probe['y'].astype("float32")
testing_Probe_y = testing_Probe_y_Org
testing_Gallery_x_Org = Gallery['x'].astype("float32") / 255.0
testing_Gallery_x = testing_Gallery_x_Org
testing_Gallery_y_Org = Gallery['y'].astype("float32")
testing_Gallery_y = testing_Gallery_y_Org
# %% Create Figures
createFig(testing_Gallery_x)
createFig(testing_Probe_x)
createFig(training_x)

# %%
# Flatten Arrays

Flatten_trainx = training_x.flatten().reshape(5933, 24576)
Flatten_testingGalleryx = testing_Gallery_x.flatten().reshape(301, 24576)
Flatten_testingProbex = testing_Probe_x.flatten().reshape(301, 24576)
print("Done")

# %%
pca = decomposition.PCA()
pca.fit(Flatten_trainx)
transformed = pca.transform(Flatten_trainx)
transformed_test_Gallery = pca.transform(Flatten_testingGalleryx)
transformed_test_Probe = pca.transform(Flatten_testingProbex)

cumulative_sum = numpy.cumsum(pca.explained_variance_ratio_, axis=0)
top95 = numpy.where(cumulative_sum > 0.95)[0][0]

fig = plt.figure(figsize=[20, 5])
count = 0
for i in range(5):
    for j in range(2):
        ax = fig.add_subplot(2, 10, count*2 + 1)
        ax.imshow(numpy.reshape(
            Flatten_trainx[count, :] - pca.mean_, (128, 64, 3)))
        ax.set_title('Original')
        ax = fig.add_subplot(2, 10, count*2 + 2)
        pca.mean_
        ax.imshow(numpy.reshape(pca.components_[0:top95, :].transpose().dot(
            numpy.reshape(transformed[count, 0:top95], (-1, 1))), (128, 64, 3)))
        ax.set_title('95% Reconstruction')
        count += 1


#%%
#transformed // testing
#transformed_test_Gallery // Test probe
#transformed_test_Probe // Test Probe
# training_y
# testing_Probe_y
# testing_Gallery_y
#testing_Probe_y.shape

print(transformed_test_Probe)
#%%
dist = distance.cdist(transformed_test_Probe, transformed_test_Gallery, 'euclidean')
num_ids = len(numpy.unique(testing_Gallery_y))
ranked_histogram = numpy.zeros(num_ids)
for i in range(len(testing_Gallery_y)):
    #print("True: {}".format(testing_Gallery_y[i]))
    order = numpy.argsort(dist[i])
    ranked = testing_Gallery_y[order]
    ranked_result = numpy.where(ranked == testing_Probe_y[i])[0][0]
    ranked_histogram[ranked_result] += 1
    #print(ranked_result)
      
# print(ranked_histogram)

plt.plot(ranked_histogram)

#%%
cmc = numpy.zeros(num_ids)
for i in range(num_ids):
    cmc[i] = numpy.sum(ranked_histogram[:(i + 1)])
fig = plt.figure(figsize=[20, 8])
ax = fig.add_subplot(1, 2, 1)
ax.plot(cmc)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('CMC Curve')
ax = fig.add_subplot(1, 2, 2)
ax.plot(cmc/num_ids)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('Percentage CMC Curve')

#%%
print("Top 1 performance: {}".format(cmc[0]/num_ids))
print("Top 5 performance: {}".format(cmc[4]/num_ids))
print("Top 10 performance: {}".format(cmc[9]/num_ids))
print("Top 100 performance: {}".format(cmc[99]/num_ids))

# %% Deep learning  Method Helper functions
def GetSiameseData(imgs, labels, batch_size):
    image_a = numpy.zeros((batch_size, numpy.shape(imgs)[1], numpy.shape(imgs)[2], numpy.shape(imgs)[3]));
    image_b = numpy.zeros((batch_size, numpy.shape(imgs)[1], numpy.shape(imgs)[2], numpy.shape(imgs)[3]));
    label = numpy.zeros(batch_size);
    for i in range(batch_size):
        if (i % 2 == 0):
            idx1 = random.randint(0, len(imgs) - 1)
            idx2 = random.randint(0, len(imgs) - 1)
            l = 1
            while (labels[idx1] != labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)            
        else:
            idx1 = random.randint(0, len(imgs) - 1)
            idx2 = random.randint(0, len(imgs) - 1)
            l = 0
            while (labels[idx1] == labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)
        image_a[i, :, :, :] = imgs[idx1,:,:,:]
        image_b[i, :, :, :] = imgs[idx2,:,:,:]
        label[i] = l
    return [image_a, image_b], label

def PairGenerator(imgs, labels, batch_size):
    while True:
        [image_a, image_b], label = GetSiameseData(imgs, labels, batch_size)
        yield [image_a, image_b], label

def conv_block(inputs, filters, spatial_dropout = 0.0, max_pool = True):
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (spatial_dropout > 0.0):
        x = layers.SpatialDropout2D(spatial_dropout)(x)
    if (max_pool == True):
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
    return x

def fc_block(inputs, size, dropout):
    x = layers.Dense(size, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (dropout > 0.0):
        x = layers.Dropout(dropout)(x)
    return x

def vgg_net(inputs, filters, fc, spatial_dropout = 0.0, dropout = 0.0):
    x = inputs
    for idx,i in enumerate(filters):
        x = conv_block(x, i, spatial_dropout, not (idx==len(filters) - 1))
    x = layers.Flatten()(x)
    for i in fc:
        x = fc_block(x, i, dropout)
    return x


#%%
deepLearning_x_train = training_x_Org.reshape(training_x_Org.shape[0], 128, 64, 3)
deepLearning_y_train = training_y_Org.reshape(training_y_Org.shape[0], 1)
deepLearning_x_test = testing_Gallery_x_Org.reshape(testing_Gallery_x_Org.shape[0], 128, 64, 3)
deepLearning_y_test = testing_Gallery_y_Org.reshape(testing_Gallery_y_Org.shape[0], 1)
#%%
test = PairGenerator(deepLearning_x_train, deepLearning_y_train, 20)
x, y = next(test)
print(y)
fig = plt.figure(figsize=[25, 6])
for i in range(10):
    ax = fig.add_subplot(2, 10, i*2 + 1)
    ax.imshow(x[0][i,:,:,0])
    ax.set_title('Pair ' + str(i) +'; Label: ' + str(y[i]))
    ax = fig.add_subplot(2, 10, i*2 + 2)
    ax.imshow(x[1][i,:,:,0])    
    ax.set_title('Pair ' + str(i) +'; Label: ' + str(y[i])) 

#%%
embedding_size = 32
dummy_input = keras.Input((128, 64, 3))
base_network = vgg_net(dummy_input, [8, 16, 32], [256], 0.2, 0)
embedding_layer = layers.Dense(embedding_size, activation=None)(base_network)
base_network = keras.Model(dummy_input, embedding_layer, name='SiameseBranch')
base_network.summary()

#%%
input_a = keras.Input((128, 64, 3), name='InputA')
input_b = keras.Input((128, 64, 3), name='InputB')

embedding_a = base_network(input_a)
embedding_b = base_network(input_b)
# %%
combined = layers.concatenate([embedding_a, embedding_b])
combined = layers.Dense(128, activation='relu')(combined)
output = layers.Dense(1, activation='sigmoid')(combined)
siamese_network = keras.Model([input_a, input_b], output, name='SiameseNetwork')
siamese_network.summary()

# %%
keras.utils.plot_model(siamese_network, show_shapes=True)
#%%
siamese_network.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
# %%
batch_size = 64
training_gen = PairGenerator(deepLearning_x_train, deepLearning_y_train, batch_size)
siamese_test_x, siamese_test_y = GetSiameseData(deepLearning_x_test, deepLearning_y_test, 10000)
siamese_network.fit(training_gen, steps_per_epoch = 30000 // batch_size, epochs=10, validation_data = (siamese_test_x, siamese_test_y))

#%%
x, y = GetSiameseData(deepLearning_x_test, deepLearning_y_test, 10)
res = siamese_network.predict(x)

fig = plt.figure(figsize=[25, 6])
for i in range(10):
    ax = fig.add_subplot(2, 10, i*2 + 1)
    ax.imshow(x[0][i,:,:,0])
    ax.set_title('Pair ' + str(i) +'; Label: ' + str(y[i]))
    
    ax = fig.add_subplot(2, 10, i*2 + 2)
    ax.imshow(x[1][i,:,:,0])    
    ax.set_title('Predicted: ' + str(res[i]))
# %%
for i in range(10):
    x[0][i,:] = x[0][0,:]

res = siamese_network.predict(x)

fig = plt.figure(figsize=[25, 6])
for i in range(10):
    ax = fig.add_subplot(2, 10, i*2 + 1)
    ax.imshow(x[0][i,:,:,0])
    ax.set_title('Pair ' + str(i))
    
    ax = fig.add_subplot(2, 10, i*2 + 2)
    ax.imshow(x[1][i,:,:,0])    
    ax.set_title('Predicted: ' + str(res[i]))
# %%
print(deepLearning_y_test)
embeddings = base_network.predict(deepLearning_x_test)
tsne_embeddings = TSNE(random_state=4).fit_transform(embeddings)
fig = plt.figure(figsize=[12, 12])
ax = fig.add_subplot(1, 1, 1)
ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c = deepLearning_y_test.flatten());
# %% CNN
dummy_input = keras.Input((128, 64, 3))
vgg_network = vgg_net(dummy_input, [8, 16, 32], [256], 0.2, 0)
embedding_layer = layers.Dense(embedding_size, activation=None)(vgg_network)
output_layer = layers.Dense(10, activation=None, name='feature_extractor')(embedding_layer)
vgg_network = keras.Model(dummy_input, output_layer, name='SimpleVGGNetwork')

vgg_network.summary()

#%%


#%%
embeddings = base_network.predict(deepLearning_x_test)
tsne_embeddings = TSNE(random_state=4).fit_transform(embeddings)
fig = plt.figure(figsize=[12, 12])
ax = fig.add_subplot(1, 1, 1)
ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c = deepLearning_y_test.flatten());


#%%
vgg_network.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = vgg_network.fit(testing_Gallery_x_Org, testing_Gallery_y_Org,
                        batch_size=64,
                        epochs=10,
                        validation_data = (deepLearning_x_test, deepLearning_y_test))

# %%
intermediate_layer_model = keras.Model(inputs=vgg_network.input,
                                       outputs=vgg_network.get_layer('feature_extractor').output)
embeddings = intermediate_layer_model(deepLearning_x_test)
tsne_embeddings = TSNE(random_state=4).fit_transform(embeddings)
fig = plt.figure(figsize=[12, 12])
ax = fig.add_subplot(1, 1, 1)
ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c = deepLearning_y_test.flatten());
# %%
# %%
embeddingGallery = base_network.predict(testing_Gallery_x_Org)
embeddingProbe = base_network.predict(testing_Probe_x_Org)

dist = distance.cdist(embeddingProbe,
                      embeddingGallery, 'euclidean')
num_ids = len(numpy.unique(testing_Gallery_y))
ranked_histogram = numpy.zeros(num_ids)
for i in range(len(testing_Gallery_y)):
    #print("True: {}".format(testing_Gallery_y[i]))
    order = numpy.argsort(dist[i])
    ranked = testing_Gallery_y[order]
    ranked_result = numpy.where(ranked == testing_Probe_y[i])[0][0]
    ranked_histogram[ranked_result] += 1
    # print(ranked_result)

print(ranked_histogram)
plt.plot(ranked_histogram)

cmc = numpy.zeros(num_ids)
for i in range(num_ids):
    cmc[i] = numpy.sum(ranked_histogram[:(i + 1)])
fig = plt.figure(figsize=[20, 8])
ax = fig.add_subplot(1, 2, 1)
ax.plot(cmc)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('CMC Curve')
ax = fig.add_subplot(1, 2, 2)
ax.plot(cmc/num_ids)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('Percentage CMC Curve')

#%%
print("Top 1 performance: {}".format(cmc[0]/num_ids))
print("Top 5 performance: {}".format(cmc[4]/num_ids))
print("Top 10 performance: {}".format(cmc[9]/num_ids))
print("Top 100 performance: {}".format(cmc[99]/num_ids))
# %%
print(cmc/num_ids)
# %%
