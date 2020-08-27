# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:51:39 2020

@author: shrin
"""
#Import required libraries
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#import CIFAR-10 Dataset from Keras
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#Regularize the images, such that the scale from 0-255 is reduiced
train_images = train_images / 255
test_images = test_images / 255

model = keras.Sequential([
        #Network topology is standard for convnets, alternating 2D Convolution layers
        #and MaxPooling layers. The number of feature maps increases as tensors are
        #passed from one layer to the next to analyze larger image patterns. 
        #MaxPool layers used to downsample feature maps by extracting the maximum
        #value of each channel. 
        #Dropout layers used to avoid overfitting, requires extra epochs to train to accuracy.
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)), #First conv layer
        keras.layers.MaxPool2D((2,2)),  #First MaxPool layer
        keras.layers.Dropout(0.2), #First dropout layer
        keras.layers.Conv2D(64, (3,3), activation='relu'), 
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(), #Tensors flattened to 1D to map to a dense layer.
        keras.layers.Dense(10, activation='softmax') #Dense layer with 10 feature maps representing 10 classes.
    ])

#Compile model using the adam optimizer, SparseCategoricalCrossEntropy loss function.
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Train the model over 15 epochs, extra epochs to account for dropout layers.
history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))
#Save the model
model.save('CIFAR10Model')


# Visualize history
# Plot history: Loss
plt.plot(history.history['loss'])
plt.title('Training loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['accuracy'])
plt.title('Training accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()
 
# Plot history: Validation Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()