
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

def getSimpleMotionModelOnly(input_shape, n_classes, printmod=1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))

    if (printmod==1 ):
        model.summary()
    return model

def getKerasCifarMotionModelOnly(input_shape, n_classes, printmod=1):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    if (printmod==1 ):
        model.summary()
    return model

def getVggMotionModel(input_shape, printmod=1):
    
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


def getKerasCifarMotionModel2(input_shape, n_classes, printmod=1, dropout=1):
    model = Sequential()
    weight_decay = 1e-4
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    if dropout:
        model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    if dropout:
        model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    if dropout:
        model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    if dropout:
        model.add(Dropout(0.5))
    
    model.add(Dense(n_classes, activation='softmax'))
                         
    if (printmod==1 ):
        model.summary()
    return model


def getSimonyanOxfordModel(input_shape, n_classes, printmod=1, dropout=1):
    model = Sequential()
    weight_decay = 1e-4
    model.add(Conv2D(96, (7,7), strides=2, padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, (5,5), strides=2, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))
    
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.9))      
    
    model.add(Dense(n_classes, activation='softmax'))
                         
    if (printmod==1 ):
        model.summary()
    
    return model

def getVggMotionModel2(input_shape, printmod=1):
    

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.get_layer('block5_pool').output   # collect outputs from hidden layer, Block 5
    # stitch layers to the VGG16 layers
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.9)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.9)(x)

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