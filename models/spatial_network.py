
import random
import cv2, numpy as np
import pickle
import csv


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras import regularizers
from keras.callbacks import History 
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Input

def getMotionModel(LR, input_shape, n_classes, printmod=1):

    img_input = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='random_uniform')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='random_uniform')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='random_uniform')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='random_uniform')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='random_uniform')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='random_uniform')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='random_uniform')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='random_uniform')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=img_input, outputs=predictions, name='vgg16_motion_model')
            
    mypotim = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    
    if (printmod==1 ):
        model.summary()
    return model

def getVggModel(input_shape, printmod=1):

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.get_layer('block5_pool').output   # collect outputs from hidden layer, Block 5
    # stitch layers to the VGG16 layers
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # freeze original VGG16 layers
    for i, layer in enumerate(model.layers):
        if 'block1' in layer.name or 'block2' in layer.name or 'block3' in layer.name or \
        'block4' in layer.name or 'block5' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    if (printmod==1 ):
        model.summary()
    return model

