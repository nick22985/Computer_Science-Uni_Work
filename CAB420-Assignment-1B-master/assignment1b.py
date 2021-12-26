#%%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import scipy.io
import numpy
#%% Importing data
q1_train = scipy.io.loadmat(('./Data/Q1/q1_train.mat'))
q1_test = scipy.io.loadmat(('./Data/Q1/q1_test.mat'))
x_train = q1_train["train_X"].astype('float32') /255.0
y_train = q1_train["train_Y"]
x_test = q1_test["test_X"].astype('float32')/255.0
y_test = q1_test["test_Y"]

y_train -= 1
y_test -= 1

#%%
x_train = numpy.transpose(x_train, [3, 0, 1, 2])
x_test = numpy.transpose(x_test, [3, 0, 1, 2])
#%%
q1_train.keys()
#%%
q1_test.keys()
#%%
print("Train")
print("X: ", x_train.shape)
print("Y: ", y_train.shape)

#%% Printing Data
fig = plt.figure(figsize=[50, 50])
for i in range(10):
    ax = fig.add_subplot(50, 50, i + 1)
    ax.imshow(x_train[:,:,:,i])

#%%
def eval_model(model, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    i = tf.cast([], tf.int32)
    indexes = tf.gather_nd(indexes, i)

    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)


# %%
inputs = keras.Input(shape=(32, 32, 3, ), name='img')
x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(inputs)
x = layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='simple_cifar_cnn')
    
def plot_training(history, model, x_test, y_test):
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(history.history['loss'], label="Training Loss")
    ax.plot(history.history['val_loss'], label="Validation Loss")
    ax.legend()

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(history.history['accuracy'], label="Training Accuracy")
    ax.plot(history.history['val_accuracy'], label="Validation Accuracy")
    ax.legend();
    
    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)
    i = tf.cast([], tf.int32)
    indexes = tf.gather_nd(indexes, i)
    
    cm = confusion_matrix(y_test, indexes)
    ax = fig.add_subplot(1, 3, 3)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)

# %%
model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
history = model_cnn.fit(x_train, y_train,
                        batch_size=64,
                        epochs=20,
                        verbose=False)
plot_training(history, model_cnn, x_test, y_test)
# %%
